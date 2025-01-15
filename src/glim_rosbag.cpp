#include <glob.h>
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
#include <boost/format.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_compression/sequential_compression_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <glim/util/config.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim_ros/glim_ros.hpp>
#include <glim_ros/ros_compatibility.hpp>

/////////
//for reconstruction part
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glk/io/ply_io.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <fstream>
//..................
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/impl/integrated_icp_factor_impl.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <iomanip>

#include <glim/util/raw_points.hpp>

////////////////////////////
//Save to LAS PDAL library headers
#include <pdal/PointView.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/StageFactory.hpp>
#include <io/BufferReader.hpp>
////////////////////////////
#include <glim/common/cloud_covariance_estimation.hpp>
///////////


class SpeedCounter {
public:
  SpeedCounter() : last_sim_time(0.0), last_real_time(std::chrono::high_resolution_clock::now()) {}

  void update(const double& stamp) {
    const auto now = std::chrono::high_resolution_clock::now();
    if (now - last_real_time < std::chrono::seconds(5)) {
      return;
    }

    if (last_sim_time > 0.0) {
      const auto real = now - last_real_time;
      const auto sim = stamp - last_sim_time;
      const double playback_speed = sim / (std::chrono::duration_cast<std::chrono::nanoseconds>(real).count() / 1e9);
      spdlog::info("playback speed: {:.3f}x", playback_speed);
    }

    last_sim_time = stamp;
    last_real_time = now;
  }

private:
  double last_sim_time;
  std::chrono::high_resolution_clock::time_point last_real_time;
};

/////////////////////////////////////////////////////////////////////
//Reconstruction part begin

//utilities

//DEBUG совсем примитивный принт точек
void print_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
{
  sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_intensity(*msg, "intensity");
  static std::size_t msg_num;
  std::size_t counter = 0, empty_coords_counter = 0,
              intens_counter_10 = 0, intens_counter_30 = 0,
              intens_counter_50 = 0, intens_counter_good = 0;

  for(size_t i = 0; i < msg->height * msg->width; ++i, ++iter_x, ++iter_y, ++iter_z) {
    //std::cout << "coords: " << *iter_x << ", " << *iter_y << ", " << *iter_z << '\n';
    counter++;
    if(*iter_x == 0.0 && *iter_y == 0.0 && *iter_z == 0.0)
    {
      empty_coords_counter++;
    }

    if(*iter_intensity < 10.0)
    {
      intens_counter_10++;
    }

    if((*iter_intensity > 10.0) && (*iter_intensity < 30.0))
    {
      intens_counter_30++;
    }


    if((*iter_intensity > 30.0) && (*iter_intensity < 50.0))
    {
      intens_counter_50++;
    }

    if((*iter_intensity > 50.0))
    {
      intens_counter_good++;
    }
  }

  msg_num++;
  std::cout << "msg_num = " << msg_num 
  << " time = " << msg->header.stamp.sec << "." << msg->header.stamp.nanosec
  << ", " << " counter = " << counter
  << ", " << " empty_coords_counter = " << empty_coords_counter
  << ", " << " intens_counter_10 = " << intens_counter_10 
  << ", " << " intens_counter_30 = " << intens_counter_30 
  << ", " << " intens_counter_50 = " << intens_counter_50 
  << ", " << " intens_counter_good = " << intens_counter_good 
  << '\n';
  
}

std::ostream& operator << (std::ostream& os, const Eigen::Isometry3d& rhs)                                                                                  //for struct output
{
   const Eigen::Quaterniond quat(rhs.linear());
   const Eigen::Vector3d trans(rhs.translation());
   os << std::setprecision(15) 
      << trans.x() << '\t' 
      << trans.y() << '\t' 
      << trans.z() << '\t'
      << quat.x()  << '\t'
      << quat.y() << '\t'
      << quat.z() << '\t'
      << quat.w() << '\t'
      << std::endl;
    return os;
}

bool operator== (const Eigen::Isometry3d& lhs, const Eigen::Isometry3d& rhs)                                                                                  //for struct output
{
   const double epsilon = 0.01; 
   const Eigen::Quaterniond quat_rhs(rhs.linear());
   const Eigen::Vector3d trans_rhs(rhs.translation());
   const Eigen::Quaterniond quat_lhs(lhs.linear());
   const Eigen::Vector3d trans_lhs(lhs.translation());
   return (fabs(trans_lhs.x() - trans_rhs.x()) < epsilon)&&
            (fabs(trans_lhs.y() -  trans_rhs.y()) < epsilon)&&
            (fabs(trans_lhs.z() -  trans_rhs.z()) < epsilon)&&
            (fabs(quat_rhs.x()  -  quat_lhs.x()) < epsilon)&&
            (fabs(quat_rhs.y()  -  quat_lhs.y()) < epsilon)&&
            (fabs(quat_rhs.z()  -  quat_lhs.z()) < epsilon)&&
            (fabs(quat_rhs.w()  -  quat_lhs.w()) < epsilon);
}

//Interval map class
// интервальный map c данными из traj_lidar.txt
template<typename K, typename V>
class interval_map {
	std::map<K, V> map_;

public:
	interval_map(V const& val) {
		map_.insert(map_.end(), std::make_pair(std::numeric_limits<K>::lowest(), val));
	}

	void assign(K const& keyBegin, K const& keyEnd, V const& val) {
		if (!(keyBegin < keyEnd)) return;

		//using mapitr_t = typename decltype(map_)::iterator;

		// End of the input range
		auto itEnd = map_.find(keyEnd);
        /*
        lower_bound : Returns an iterator pointing
        to the first element that is not less than (i.e. greater or equal to) key.
        */
		if (auto l = map_.lower_bound(keyEnd); itEnd != map_.end())
			itEnd->second = l->second;
		else
			itEnd = map_.insert(map_.end(), std::make_pair(keyEnd, (--l)->second));

		// Beginning of the input range
		auto itBegin = map_.insert_or_assign(keyBegin, val).first;

		// Cleanup the new range
        /*
        template< class InputIt >
        InputIt next( InputIt it, typename std::iterator_traits<InputIt>::difference_type n = 1 );
        Return the nth successor (or -nth predecessor if n is negative) of iterator it.
        */
		map_.erase(std::next(itBegin), itEnd);

		// Make canonical
        /*
        The representation in the std::map must be canonical, that is,
         consecutive map entries must not have the same value:
         ..., (0,'A'), (3,'A'), ... is not allowed. 
        */
		auto itRight = itEnd;
		auto itLeft = (itBegin != map_.begin() ? std::prev(itBegin) : itBegin);
		while (itRight != itLeft) {
            /*
            template< class BidirIt >
            BidirIt prev( BidirIt it, typename std::iterator_traits<BidirIt>::difference_type n = 1 );
            Return the nth predecessor (or -nth successor if n is negative) of iterator it
            */
			auto itNext = std::prev(itRight);
			if (itRight->second == itNext->second)
				map_.erase(itRight);
			itRight = itNext;
		}
	}

	V const& operator[](K const& key) const {
		return (--map_.upper_bound(key))->second;
	}

	void print() {
		for (auto&&[key, val] : map_)
			std::cout << "[" << key << ':' <<  val << "]";
		std::cout << '\n';
	}

    std::size_t size() const
    {
        return map_.size();
    }
};

 
    //выбираем точки, находящиеся на расстоянии не более max_sq_dist from keyframe points
    namespace gtsam_points {
    namespace frame {

    // In this example, we show how to directly feed a custom class (in this case, std::vector<Eigen::Vector4d>) to the ICP factor.
    // You need to first define methods to access the point data in your custom class by specializing gtsam_points::frame::traits<>
    template <>
    struct traits<std::vector<Eigen::Vector4d>> {
      using T = std::vector<Eigen::Vector4d>;

      // To use the conventional ICP, you need to at least define the following methods:
      // - size(const T& frame)            : Get the number of points (This is required for only source fame)
      // - has_points(const T& frame)      : Check if your custom class has point data
      // - point(const T& frame, size_t i) : Get i-th point from your class
      static int size(const T& points) { return points.size(); }
      static bool has_points(const T& points) { return !points.empty(); }
      static const Eigen::Vector4d& point(const T& points, size_t i) { return points[i]; }
    };

    }  // namespace frame
    }  // namespace gtsam_points

    namespace std {
      template<> struct hash<Eigen::Vector4d>
      {
        std::size_t	operator()(const Eigen::Vector4d& p) const noexcept
        {
          return std::hash<double>()(p.x())
                    ^ std::hash<double>()(p.y()) 
                    ^ std::hash<double>()(p.z());
        }
      };
    }
    
std::random_device rd;  // a seed source for the random number engine
std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

//сквозная структура - заполняется по ходу 
 struct PointEigen
  {
    PointEigen(){
      time = 0.0, point_coord = Eigen::Vector4d{0,0,0,1}, intensity = 0.0, point_normal = Eigen::Vector4d{0,0,0,1};
    }
    double time;
    Eigen::Vector4d point_coord;
    double intensity;
    Eigen::Vector4d point_normal;
  };

  //DEBUG
  void save_coords_to_ply(const std::vector<PointEigen>& pev, const std::string file_name)
  {
    std::vector<Eigen::Vector4d> coords_vect;
    for (auto &&p : pev)
    {
      coords_vect.push_back(p.point_coord);
    }
    glk::save_ply_binary("/tmp/" + file_name, coords_vect.data(), coords_vect.size());
    std::cout  << "saved in " << "/tmp/" + file_name << std::endl;
  }
  //DEBUG

  std::vector<Eigen::Vector4d> get_coords(const std::vector<PointEigen>& pev)
  {
    std::vector<Eigen::Vector4d> coords_vect;
    for (auto &&p : pev)
    {
      coords_vect.push_back(p.point_coord);
    }
    return coords_vect;
  }

class Reconstructor
{
  public:
    //config params
    double par_distance_near_thresh;
    double par_distance_far_thresh;
    bool par_use_random_grid_downsampling;
    double par_downsample_resolution;
    double par_downsample_target;
    double par_downsample_rate;
    double par_enable_outlier_removal;
    int par_outlier_removal_k;
    double par_outlier_std_mul_factor;
    int par_k_correspondences;//used as k_neighbors in condense_neighbors()
    int par_num_threads;
    double par_max_sq_dist;
    double par_deviation;
    std::string par_dump_dir;
    std::string par_output_file_path;

    bool init()
    {
      //config params init
      glim::Config config(glim::GlobalConfig::get_config_path("config_reconstruct"));
      par_distance_near_thresh = config.param<double>("config_reconstruct", "distance_near_thresh", 1.0);
      par_distance_far_thresh = config.param<double>("config_reconstruct", "distance_far_thresh", 100.0);
      par_use_random_grid_downsampling = config.param<bool>("config_reconstruct", "use_random_grid_downsampling", false);
      par_downsample_resolution = config.param<double>("config_reconstruct", "downsample_resolution", 0.15);
      par_downsample_target = config.param<double>("config_reconstruct", "random_downsample_target", 0);
      par_downsample_rate = config.param<double>("config_reconstruct", "random_downsample_rate", 0.3);
      par_enable_outlier_removal = config.param<bool>("config_reconstruct", "enable_outlier_removal", false);
      par_outlier_removal_k = config.param<int>("config_reconstruct", "outlier_removal_k", 10);
      par_outlier_std_mul_factor = config.param<double>("config_reconstruct", "outlier_std_mul_factor", 2.0);
      par_k_correspondences = config.param<int>("config_reconstruct", "k_correspondences", 8);
      par_num_threads = config.param<int>("config_reconstruct", "num_threads", 2);
      par_max_sq_dist = config.param<double>("config_reconstruct", "max_sq_dist", 1.0);
      par_deviation = config.param<double>("config_reconstruct", "deviation", 0.1);
      par_dump_dir = config.param<std::string>("config_reconstruct", "dump_dir", "/tmp/dump/");//NB: Not config but CLI param in GLIM
      par_output_file_path = config.param<std::string>("config_reconstruct", "output_file_path", "/tmp/reconstructed.ply");;

      if(!fillIntervalTrajMap(par_dump_dir + std::string{"traj_lidar.txt"}))
      {
        return false;
      }

      if(!fill_keyframes_data(par_dump_dir))
        return false;

      return true;  
    }

    bool do_reconstruct()
    {
      //preprocessing
      std::vector<gtsam_points::PointCloud::Ptr> preprocessed_vect;
      for (auto &&in : raw_points_vector)
      {
        auto out = preprocess_impl(in);
        preprocessed_vect.push_back(out);
      }

      //transform input points
      for (auto &&frame : preprocessed_vect)
      {
        for(int i = 0;i<frame->size();i++)
        {
          auto T = interval_traj_map[*(frame->times + i)/1e9];//?????? for mid360 only
          auto p = *(frame->points + i);
          Eigen::Vector4d t = T*p;
          *(frame->points + i) = t;
        }
      }

     
      std::vector<PointEigen> input_points;//!!!
      for (auto &&frame : preprocessed_vect)
      {
        for(int i = 0;i<frame->size();i++)
        {
            PointEigen pe;
            if(!frame->has_points())
            {
               spdlog::warn("Err : the point cloud has not points");
               return false;
            }
            pe.point_coord = *(frame->points + i);

            if(!frame->has_intensities())
            {
               spdlog::warn("Err : the point cloud has not intesities");
               return false;
            }
            pe.intensity = *(frame->intensities + i);

            if(!frame->has_times())
            {
               spdlog::warn("Err : the point cloud has not per-point timestamps");
               return false;
            }
            pe.time = *(frame->times + i);

            input_points.push_back(pe);
        }
      }

      spdlog::warn("Reconstruction preparing. Please wait...");
      
      const double max_sq_dist = par_max_sq_dist;//1;//squared!!!
      const double deviation = par_deviation;//0.01;//linear on coord
      std::vector<PointEigen> summary_condensed_points;
      if(!condense_points(key_frame_coords,
                     input_points,
                     summary_condensed_points,
                     max_sq_dist, deviation))
      {
        spdlog::error("Condense points error");
        return false;
      }                     

      //DEBUG
      auto summary_condensed_coords = get_coords(summary_condensed_points);
      glk::save_ply_binary(par_output_file_path, summary_condensed_coords.data(), summary_condensed_coords.size());
      spdlog::warn("PLY : Reconstruction saved in {}", par_output_file_path);
      /////////////////
      const std::string las_file_name = "/tmp/summary.las";//TODO param in config
      if(!save_las(las_file_name, summary_condensed_points))
      {
        spdlog::error("Save to LAS error!");
        return false;
      }

      spdlog::warn("LAS : Reconstruction saved in {}", las_file_name);

      return true;    
    }

//...................................................................
    // интервальный map c данными из traj_lidar.txt
    using IsoMap = interval_map<double, Eigen::Isometry3d>;//indexed with timestamp intervals
    IsoMap interval_traj_map{Eigen::Isometry3d{}};//indexed by timestamp, ctor required!!
    //заполнение интервальный map c данными из traj_lidar.txt
    bool fillIntervalTrajMap(const std::string& filename)//"/tmp/dump/traj_lidar.txt"
    {
      //interval_traj_map.print();
      const size_t line_size = 200; 
      size_t lines_count = 0;
      char line[line_size];

      std::ifstream rfile;
      double pre_stamp = -1.0;
      rfile.open(filename);
      if (rfile.is_open()) {
          while (rfile.getline(line, line_size)) {
            lines_count++;
            Eigen::Isometry3d pose;
            Eigen::Quaterniond quat;
            Eigen::Vector3d trans;
            std::istringstream is(line);
            double stamp;
            is >> stamp >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();
            pose.linear() = quat.matrix();
            pose.translation() = trans;
            interval_traj_map.assign(pre_stamp, stamp, pose);
            pre_stamp = stamp;

          }
          rfile.close();
      }
      else
      {
          spdlog::error("Error: traj_lidar.txt is not open!");
          return false;
      }
      spdlog::info(" traj_lidar.txt lines_count = {}", lines_count);
      //interval_traj_map.print();
      return true;
    }//fillIntervalTrajMap

//...................................................................

    // читаем результаты glim from dump dir;
    //based on 
    //GlobalMapping::load(const std::string& path) {
    //SubMap::Ptr SubMap::load(const std::string& path) {

    struct submap_data
    {
        std::vector<double> timestamps;
        std::vector<Eigen::Vector4d> submap_points;
    };
    struct keyframes_data
    {
      int num_submaps;
      int num_all_frames; 
      std::vector<submap_data> sds;
    };
    keyframes_data kd;

    template <int ROWS, int COLS>
    Eigen::Matrix<double, ROWS, COLS> read_matrix(std::ifstream& ifs) {
      Eigen::Matrix<double, ROWS, COLS> m;
      for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
          ifs >> m(i, j);
        }
      }
      return m;
    }

    //std::vector<Eigen::Vector4d> key_frame_coords;
    //std::map<double,Eigen::Vector4d> key_frame_coords;
    std::vector<std::pair<double,Eigen::Vector4d>> key_frame_coords;

    bool fill_keyframes_data(const std::string& dump_dir_path)
    {
      std::ifstream ifs(dump_dir_path + "/graph.txt");
      if (!ifs) {
        spdlog::error("fill_keyframes_data : failed to open graph.txt");
        return false;
      }

      std::string token;
      int num_matching_cost_factors;//not used

      ifs >> token >> kd.num_submaps;
      ifs >> token >> kd.num_all_frames;
      ifs >> token >> num_matching_cost_factors;

      int num_all_frames = 0;
      for (int i = 0; i < kd.num_submaps; i++) {
        submap_data sd;
        const std::string submap_path = (boost::format("%s/%06d") % dump_dir_path % i).str();
        std::ifstream ifs(submap_path + "/data.txt");
        if (!ifs) {
          spdlog::error("failed to open data.txt");
          return false;
        }
   
        std::string token;
        int id;
        ifs >> token >> id;
 
        Eigen::Isometry3d T_world_origin;       ///< frame[frame.size() / 2] pose w.r.t. the world
        Eigen::Isometry3d T_origin_endpoint_L;  ///< frame.front() pose w.r.t. the origin
        Eigen::Isometry3d T_origin_endpoint_R;  ///< frame.back() pose w.r.t. the origin

        ifs >> token;
        T_world_origin.matrix() = read_matrix<4, 4>(ifs);//for recalculation
        ifs >> token;
        T_origin_endpoint_L.matrix() = read_matrix<4, 4>(ifs);
        ifs >> token;
        T_origin_endpoint_R.matrix() = read_matrix<4, 4>(ifs);

        ifs >> token;
        Eigen::Isometry3d T_lidar_imu(read_matrix<4, 4>(ifs));
        ifs >> token;
        Eigen::Matrix<double, 6, 1> imu_bias = read_matrix<6, 1>(ifs);

        int frame_id;
        ifs >> token >> frame_id;

        int num_frames;
        int count = 0;
        ifs >> token >> num_frames;
        num_all_frames += num_frames;

        for (int i = 0; i < num_frames; i++) {
          int id;
          double stamp;
          count = 0;
          while(token != "stamp:")
          {
            ifs >> token;
            count++;
            if(count > 100)
            {
              spdlog::error("fill_keyframes_data : Unexpected data.txt format (stamp)");
              return false;
            }
          }
          ifs >> stamp;
          sd.timestamps.push_back(stamp);
          ifs >> token;

        }

        //just one for submap  /covs.bin and /points.bin files reading
        auto merged_keyframe = gtsam_points::PointCloudCPU::load(submap_path);
        Eigen::Vector4d* ppoint = nullptr;
        for (size_t i = 0; i < merged_keyframe->size(); i++)
        {
          ppoint = merged_keyframe->points;
          Eigen::Vector4d point = T_world_origin*(*ppoint);
          sd.submap_points.push_back(point);
          merged_keyframe->points++;
        }
        kd.sds.push_back(sd);
      }
      
      //NB: num_all_frames == kd.num_all_frames
      std::cout << " num_all_frames = " << num_all_frames <<  " kd.num_all_frames = " << kd.num_all_frames << std::endl;
      
      for (auto &&sd : kd.sds)
      {
          //NB: sd.timestamps.size() != sd.submap_points.size()
          std::cout << " sd.timestamps.size() = "<< sd.timestamps.size() << " : sd.submap_points.size() = "<<  sd.submap_points.size() << std::endl;
          //for (auto &&p : sd.submap_points)
          for(size_t k = 0;k<sd.submap_points.size();k++)
          {
            //key_frame_coords.push_back(p);
            key_frame_coords.push_back({sd.timestamps.at(0/* !!!! */),sd.submap_points.at(k)});
          }
      } 
      return true;
    }//fill_keyframes_data


    using PointCloud2 = sensor_msgs::msg::PointCloud2;
    using PointCloud2Ptr = sensor_msgs::msg::PointCloud2::SharedPtr;
    using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;
    using PointField = sensor_msgs::msg::PointField;


    template <typename Stamp>
    double to_sec(const Stamp& stamp) {
      return stamp.sec + stamp.nanosec / 1e9;
    }

    template <typename T>
    Eigen::Vector4d get_vec4(const void* x, const void* y, const void* z) {
      return Eigen::Vector4d(*reinterpret_cast<const T*>(x), *reinterpret_cast<const T*>(y), *reinterpret_cast<const T*>(z), 1.0);
    }

    glim::RawPoints::Ptr extract_raw_points(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg, const std::string& intensity_channel = "intensity") {
      int num_points = points_msg->width * points_msg->height;

      int x_type = 0;
      int y_type = 0;
      int z_type = 0;
      int time_type = 0;  // ouster and livox
      int intensity_type = 0;
      int color_type = 0;

      int x_offset = -1;
      int y_offset = -1;
      int z_offset = -1;
      int time_offset = -1;
      int intensity_offset = -1;
      int color_offset = -1;

      std::unordered_map<std::string, std::pair<int*, int*>> fields;
      fields["x"] = std::make_pair(&x_type, &x_offset);
      fields["y"] = std::make_pair(&y_type, &y_offset);
      fields["z"] = std::make_pair(&z_type, &z_offset);
      fields["t"] = std::make_pair(&time_type, &time_offset);
      fields["time"] = std::make_pair(&time_type, &time_offset);
      fields["time_stamp"] = std::make_pair(&time_type, &time_offset);
      fields["timestamp"] = std::make_pair(&time_type, &time_offset);
      fields[intensity_channel] = std::make_pair(&intensity_type, &intensity_offset);
      fields["rgba"] = std::make_pair(&color_type, &color_offset);

      for (const auto& field : points_msg->fields) {
        auto found = fields.find(field.name);
        if (found == fields.end()) {
          continue;
        }

        *found->second.first = field.datatype;
        *found->second.second = field.offset;
      }

      if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        spdlog::warn("missing point coordinate fields");
        return nullptr;
      }

      if ((x_type != PointField::FLOAT32 && x_type != PointField::FLOAT64) || x_type != y_type || x_type != y_type) {
        spdlog::warn("unsupported points type");
        return nullptr;
      }

      auto raw_points = std::make_shared<glim::RawPoints>();

      raw_points->points.resize(num_points);

      if (x_type == PointField::FLOAT32 && y_offset == x_offset + sizeof(float) && z_offset == y_offset + sizeof(float)) {
        // Special case: contiguous 3 floats
        for (int i = 0; i < num_points; i++) {
          const auto* x_ptr = &points_msg->data[points_msg->point_step * i + x_offset];
          raw_points->points[i] << Eigen::Map<const Eigen::Vector3f>(reinterpret_cast<const float*>(x_ptr)).cast<double>(), 1.0;
        }
      } else if (x_type == PointField::FLOAT64 && y_offset == x_offset + sizeof(double) && z_offset == y_offset + sizeof(double)) {
        // Special case: contiguous 3 doubles
        for (int i = 0; i < num_points; i++) {
          const auto* x_ptr = &points_msg->data[points_msg->point_step * i + x_offset];
          raw_points->points[i] << Eigen::Map<const Eigen::Vector3d>(reinterpret_cast<const double*>(x_ptr)), 1.0;
        }
      } else {
        for (int i = 0; i < num_points; i++) {
          const auto* x_ptr = &points_msg->data[points_msg->point_step * i + x_offset];
          const auto* y_ptr = &points_msg->data[points_msg->point_step * i + y_offset];
          const auto* z_ptr = &points_msg->data[points_msg->point_step * i + z_offset];

          if (x_type == PointField::FLOAT32) {
            raw_points->points[i] = get_vec4<float>(x_ptr, y_ptr, z_ptr);
          } else {
            raw_points->points[i] = get_vec4<double>(x_ptr, y_ptr, z_ptr);
          }
        }
      }

      if (time_offset >= 0) {
        raw_points->times.resize(num_points);

        for (int i = 0; i < num_points; i++) {
          const auto* time_ptr = &points_msg->data[points_msg->point_step * i + time_offset];
          switch (time_type) {
            case PointField::UINT32:
              raw_points->times[i] = *reinterpret_cast<const uint32_t*>(time_ptr) / 1e9;
              break;
            case PointField::FLOAT32:
              raw_points->times[i] = *reinterpret_cast<const float*>(time_ptr);
              break;
            case PointField::FLOAT64:
              raw_points->times[i] = *reinterpret_cast<const double*>(time_ptr);
              break;
            default:
              spdlog::warn("unsupported time type {}", time_type);
              return nullptr;
          }
        }
      }

      if (intensity_offset >= 0) {
        raw_points->intensities.resize(num_points);

        for (int i = 0; i < num_points; i++) {
          const auto* intensity_ptr = &points_msg->data[points_msg->point_step * i + intensity_offset];
          switch (intensity_type) {
            case PointField::UINT8:
              raw_points->intensities[i] = *reinterpret_cast<const std::uint8_t*>(intensity_ptr);
              break;
            case PointField::UINT16:
              raw_points->intensities[i] = *reinterpret_cast<const std::uint16_t*>(intensity_ptr);
              break;
            case PointField::UINT32:
              raw_points->intensities[i] = *reinterpret_cast<const std::uint32_t*>(intensity_ptr);
              break;
            case PointField::FLOAT32:
              raw_points->intensities[i] = *reinterpret_cast<const float*>(intensity_ptr);
              break;
            case PointField::FLOAT64:
              raw_points->intensities[i] = *reinterpret_cast<const double*>(intensity_ptr);
              break;
            default:
              spdlog::warn("unsupported intensity type {}", intensity_type);
              return nullptr;
          }
        }
      }

      if (color_offset >= 0) {
        if (color_type != PointField::UINT32) {
          spdlog::warn("unsupported color type {}", color_type);
        } else {
          raw_points->colors.resize(num_points);

          for (int i = 0; i < num_points; i++) {
            const auto* color_ptr = &points_msg->data[points_msg->point_step * i + color_offset];
            raw_points->colors[i] = Eigen::Matrix<unsigned char, 4, 1>(reinterpret_cast<const std::uint8_t*>(color_ptr)).cast<double>() / 255.0;
          }
        }
      }

      raw_points->stamp = to_sec(points_msg->header.stamp);
      return raw_points;
    }//reconstruct_extract_raw_points

    //select preprocess options by external params
    gtsam_points::PointCloud::Ptr preprocess_impl(const glim::RawPoints::ConstPtr& raw_points)
    {
      std::mt19937 mt;
      gtsam_points::PointCloud::Ptr frame(new gtsam_points::PointCloud);
      frame->num_points = raw_points->size();
      frame->times = const_cast<double*>(raw_points->times.data());//relative times
      for(int i = 0;i<raw_points->size();i++)
      {
        *(frame->times + i) += raw_points->stamp;//unix times
      }
      
      frame->points = const_cast<Eigen::Vector4d*>(raw_points->points.data());
      if (raw_points->intensities.size()) {
        //spdlog::info(" EG: raw_points->intensities.size() !=0");
        frame->intensities = const_cast<double*>(raw_points->intensities.data());
      }

      //Downsampling
      //.........................................................................

      if(par_use_random_grid_downsampling)
      {
        auto random_downsample_target = par_downsample_target;//10000;//5000;//: Target number of points for voxel-based random sampling (Enabled when > 0)
        //when random_downsample_target is down then output size is down
        auto random_downsample_rate = par_downsample_rate;//0.1;// Sampling rate for voxel-based random sampling (Enabled when target <= 0)
        //when random_downsample_rate is changed then output size is not changed
        const double rate = random_downsample_target > 0 ? static_cast<double>(random_downsample_target) / frame->size() : random_downsample_rate;
        auto num_threads = par_num_threads;//2;
        auto downsample_resolution = par_downsample_resolution;//1.0;//0.25;// Voxel grid resolution for downsampling
        //std::cout << " randomgrid_sampling input size = " << frame->size() << std::endl;
        frame = gtsam_points::randomgrid_sampling(frame, downsample_resolution, rate, mt, num_threads);
        //std::cout << " randomgrid_sampling output size = " << frame->size() << std::endl;
      }
      else
      {
      //.........................................................................
        auto num_threads = par_num_threads;//2;
        auto downsample_resolution = par_downsample_resolution;//1.0;// Voxel grid resolution for downsampling
        frame = gtsam_points::voxelgrid_sampling(frame, downsample_resolution, num_threads);
      }
      //.........................................................................
    ///////////////////////////////////////////////////////////////////////////////////////////////////
      //Distance filter
      auto distance_near_thresh = par_distance_near_thresh;//0.5;//0.25;
      auto distance_far_thresh = par_distance_far_thresh;//100;
      std::vector<int> indices;
      indices.reserve(frame->size());
      for (int i = 0; i < frame->size(); i++) {
        const bool is_finite = frame->points[i].allFinite();
        const double dist = (Eigen::Vector4d() << frame->points[i].head<3>(), 0.0).finished().norm();
        if (dist > distance_near_thresh && dist < distance_far_thresh && is_finite) {
          indices.push_back(i);
        }
      }

      // Sort by time
      std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) { return frame->times[lhs] < frame->times[rhs]; });
      frame = gtsam_points::sample(frame, indices);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
        // Outlier removal
        auto enable_outlier_removal = par_enable_outlier_removal;
        if (enable_outlier_removal) {
          auto outlier_removal_k = par_outlier_removal_k;//10;
          auto outlier_std_mul_factor = par_outlier_std_mul_factor;//1.0;
          //num_threads
          frame = gtsam_points::remove_outliers(frame, outlier_removal_k, outlier_std_mul_factor, par_num_threads);
        }
    ////////////////////////////////////////////////////////////////
   
      return frame;
    }//reconstruct_preprocess_impl

//...................................................................
   
    //выбираем точки, находящиеся на расстоянии не более max_sq_dist from keyframe points
    //std::random_device rd;  // a seed source for the random number engine
    //std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    
    double generate_random_coord(const double kf_coord, const double deviation)
    {
      //std::random_device rd;  // a seed source for the random number engine
      //std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
      std::uniform_int_distribution<> distrib((kf_coord - deviation)*1000, (kf_coord + deviation)*1000);
      return (distrib(gen)/1000.0f);
    }

    //not used
    void condense_neighbors(const std::vector<Eigen::Vector4d>& key_frame_points,
                              std::vector<Eigen::Vector4d>& input_points,
                              std::vector<Eigen::Vector4d>& summary_condensed_points,
                              const double max_sq_dist,
                              const double deviation)
    {
      std::unordered_set<Eigen::Vector4d> neighbors_set;
      std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(input_points.data(), input_points.size()));
      int k_neighbors = par_k_correspondences;//8;
      std::vector<size_t> k_indices(k_neighbors);
      std::vector<double> k_sq_dists(k_neighbors);
      size_t nn;
    

      for (size_t k = 0; k < key_frame_points.size(); k++)
      {
          size_t nn = target_tree->knn_search(
                                  gtsam_points::frame::point(key_frame_points, k).data(),
                                  k_neighbors,
                                    k_indices.data(),
                                    k_sq_dists.data(),
                                    max_sq_dist
                                  );
          
          for (size_t i = 0; i < k_neighbors; i++)
          {
            if(k_indices[i]!=std::numeric_limits<size_t>::max())
            {
                input_points.at(k_indices[i]).x() = generate_random_coord(key_frame_points[k].x(), deviation);
                input_points.at(k_indices[i]).y() = generate_random_coord(key_frame_points[k].y(), deviation);
                input_points.at(k_indices[i]).z() = generate_random_coord(key_frame_points[k].z(), deviation);
                neighbors_set.insert(input_points.at(k_indices[i]));
            }
          }
      }

      //move from to summary_condensed_points
      summary_condensed_points.resize(neighbors_set.size());
      size_t l = 0;
      for (auto &&ns : neighbors_set)
      {
          summary_condensed_points[l++] = std::move(ns);
      }

      //add keyframes
      summary_condensed_points.insert(summary_condensed_points.end(),  key_frame_points.begin(), key_frame_points.end());   
    }//condense_neighbors




    bool condense_points(const std::vector<std::pair<double,Eigen::Vector4d>>& key_frame_coords /*const std::vector<Eigen::Vector4d>& key_frame_coords*/,
                              std::vector<PointEigen>& input_points,
                              std::vector<PointEigen>& summary_condensed_points,
                              const double max_sq_dist,
                              const double deviation)
    {
      if(key_frame_coords.empty() || input_points.empty())
      {
        spdlog::error("Sth going wrong - no input or keyframes data");
        return false;
      }
      //1) Ищем среди input_points соседние с key_frame_coords - то есть находящиеся не далее max_sq_dist от них,
      // но в кол-ве не более k_neighbors  
      //std::unordered_set<Eigen::Vector4d> neighbors_set;
      std::unordered_map<Eigen::Vector4d, PointEigen> neighbors_map;//indexed by coords
      auto input_coords = get_coords(input_points);
      std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(input_coords.data(), input_coords.size()));
      int k_neighbors = par_k_correspondences;//8;
      std::vector<size_t> k_indices(k_neighbors);
      std::vector<double> k_sq_dists(k_neighbors);
      size_t nn;

      std::vector<PointEigen> key_frame_points;
      key_frame_points.reserve(key_frame_coords.size());

      //std::cout << " max_sq_dist : "<< max_sq_dist << std::endl;
      //static double last_time;
      int alone_kf_points_counter = 0;
      
      spdlog::warn("condense_points key_frame_coords size = {}", key_frame_coords.size());

      //for (size_t k = 0; k < key_frame_coords.size(); k++)
      for (auto &&kf : key_frame_coords)
      {
          size_t nn = target_tree->knn_search(
                                  //gtsam_points::frame::point(key_frame_coords, k).data(),
                                  
                                  kf.second.data(),
                                  k_neighbors,
                                    k_indices.data(),
                                    k_sq_dists.data(),
                                    max_sq_dist
                                  );
          bool has_neighbor = false;
          for (size_t i = 0; i < k_neighbors; i++)
          {
            if(k_indices[i]!=std::numeric_limits<size_t>::max())
            {
                has_neighbor = true;//has at least one neighbor
                //input_points.at(k_indices[i]).point_coord.x() = generate_random_coord(key_frame_coords[k].x(), deviation);
                //input_points.at(k_indices[i]).point_coord.y() = generate_random_coord(key_frame_coords[k].y(), deviation);
                //input_points.at(k_indices[i]).point_coord.z() = generate_random_coord(key_frame_coords[k].z(), deviation);
                //std::cout << i << " - "<< k_indices[i] << std::endl;
                neighbors_map.insert({input_points.at(k_indices[i]).point_coord, input_points.at(k_indices[i])});
            }
          }

          //2) Достраиваем key_frame_point на основе key_frame_coord и инфы в соседях
          PointEigen key_frame_point;
          /*key_frame_point.point_coord = key_frame_coords[k];*/
          key_frame_point.point_coord = kf.second;
          if(has_neighbor)
          {
            //std::cout << " : "<< k_indices[0] << std::endl;
            key_frame_point.intensity = input_points.at(k_indices[0]).intensity;
            key_frame_point.time = input_points.at(k_indices[0]).time;
            //last_time = input_points.at(k_indices[0]).time;//???
          }
          else
          {
            //spdlog::warn("Sth going wrong - no neighbors for keyframe");//It's not impossible !!!! Why ?
            //std::cout << key_frame_point.point_coord << std::endl;
            //std::cout << std::fixed << std::setprecision(6) << "last_time : " << last_time << std::endl;
            key_frame_point.intensity = 15;//? Or ignore this point ? which % of alone kf points ? which are her coords?
            //key_frame_point.time = last_time;//?
            key_frame_point.time = kf.first;//очень приблизительно - на основе первого(!) stamp в data.txt в submap
            alone_kf_points_counter++;
          }

          key_frame_points.push_back(key_frame_point);

      }

      spdlog::warn("alone_kf_points_counter = {}",alone_kf_points_counter);
      spdlog::warn("condensed {} neighbors", neighbors_map.size());
        
      //3) move from hash map to summary_condensed_points vector
      summary_condensed_points.resize(neighbors_map.size());
      size_t l = 0;
      for (auto &&ns : neighbors_map)
      {
          summary_condensed_points[l++] = std::move(ns.second);
      }
      
      //4) add keyframes points to summary
      summary_condensed_points.insert(summary_condensed_points.end(),  key_frame_points.begin(), key_frame_points.end());

      //5) calcul and fill normals fields in points
      //TODO : k param can have diff values for kf neighbors and normals calculation
     return calculate_normals(summary_condensed_points, par_k_correspondences);

         
    }//condense_points


//...................................................................

  std::vector<std::shared_ptr<glim::RawPoints>> raw_points_vector;
  
  bool points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    auto raw_points = extract_raw_points(msg);
    if (raw_points == nullptr) {
      spdlog::warn("failed to extract points from message");
      return false;
    }
    raw_points->stamp += 0.0;//points_time_offset; 
    raw_points_vector.push_back(raw_points);
    return true;
  }

// las saving below
  bool fillView(pdal::PointViewPtr view, const std::vector<PointEigen>& points)
  {
    
    try
    {
      for (size_t l = 0; l< points.size();l ++)
      {
        if(points.at(l).time == 0.0) { spdlog::error("Reconstructor::fillView error: time == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::GpsTime, l , points.at(l).time);
        //if(points.at(l).point_coord.x() == 0.0) { spdlog::error("Reconstructor::fillView error: point_coord.x() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::X, l , points.at(l).point_coord.x());
        //if(points.at(l).point_coord.y() == 0.0) { spdlog::error("Reconstructor::fillView error: point_coord.y() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::Y, l , points.at(l).point_coord.y());
        //if(points.at(l).point_coord.z() == 0.0) { spdlog::error("Reconstructor::fillView error: point_coord.z() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::Z, l , points.at(l).point_coord.z());
        //if(points.at(l).intensity == 0.0) { spdlog::error("Reconstructor::fillView error: intensity == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::Intensity, l , points.at(l).intensity);
        //if(points.at(l).point_normal.x() == 0.0) { spdlog::error("Reconstructor::fillView error: point_normal.x() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::NormalX, l , points.at(l).point_normal.x());
        //if(points.at(l).point_normal.y() == 0.0) { spdlog::error("Reconstructor::fillView error: point_normal.y() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::NormalY, l , points.at(l).point_normal.y());
        //if(points.at(l).point_normal.z() == 0.0) { spdlog::error("Reconstructor::fillView error: point_normal.z() == 0.0 "); return false; }
        view->setField(pdal::Dimension::Id::NormalZ, l , points.at(l).point_normal.z());
      }
  
    }
    catch(const std::exception& e)
    {
      spdlog::error("Reconstructor::fillView error: {}",e.what());
      return false;
    }
    spdlog::info("Reconstructor::fillView points count = {}", points.size());
    return true;
  }

  bool save_las(const std::string& las_file_name, const std::vector<PointEigen>& points)
  {
  /*
  в LAS должны быть все данные
  XYZ, нормали (они же угол излучения), GPS-время, интенсивность
  */
  /*
  PDAL:
  In addition to the X, Y, Z coordinates of a point, PDAL can write many other attributes. Their full
  list and types are in [Dimension.json]. Most of these are not enabled by default. To enable them, the
  LAS file format minor version should be set to 4, the value of the `extra_dims` writer option should be `all`,
  and the attributes should be registered with the function ``registerDim()``.

  */

    //////////////////////////////////////////////////////////////////////
    spdlog::info("Reconstructor  saves to LAS");
    using namespace pdal;
    Options options;
    options.add("filename", las_file_name);
    options.add("extra_dims", "all");
    options.add("minor_version", 4);

    PointTable table;
    table.layout()->registerDim(Dimension::Id::GpsTime);
    table.layout()->registerDim(Dimension::Id::X);
    table.layout()->registerDim(Dimension::Id::Y);
    table.layout()->registerDim(Dimension::Id::Z);
    table.layout()->registerDim(Dimension::Id::Intensity);
    table.layout()->registerDim(Dimension::Id::NormalX);
    table.layout()->registerDim(Dimension::Id::NormalY);
    table.layout()->registerDim(Dimension::Id::NormalZ);

    spdlog::info(std::string("Writing to file : ") + las_file_name);
    PointViewPtr view(new PointView(table));
    if(!view)
    {
      spdlog::error("Unable to const PointView");
      return false;
    }

    if(!fillView(view, points))
      return false;

    BufferReader reader;
    reader.addView(view);

    StageFactory factory;

    // StageFactory always "owns" stages it creates. They'll be destroyed with the factory.
    Stage *writer = factory.createStage("writers.las");
    if(!writer)
    {
          std::cout << "Unable to create writer..." << std::endl;
          return false;
    }

    writer->setInput(reader);
    writer->setOptions(options);
    writer->prepare(table);
    writer->execute(table);
  ////////////////////////////////////////////////////////////////////////  
    return true;
  }
//las saving above


// normals calculation below
std::vector<int> find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k)  {
  gtsam_points::KdTree tree(points, num_points);

  std::vector<int> neighbors(num_points * k);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k);
    std::vector<double> k_sq_dists(k);
    tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  };

  int num_threads = 2;//temp
  
  if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not enabled" << std::endl;
    abort();
#endif
  }

  return neighbors;
}

bool calculate_normals(std::vector<PointEigen>& points, const int kn)
{
   std::vector<Eigen::Vector4d> normals;

   const int num_points = points.size();
   std::vector<Eigen::Vector4d> temp;

   for (auto &&p : points)
   {
      temp.push_back(p.point_coord);    
   }
   
   auto nn = find_neighbors(temp.data(), num_points, kn);
   if(nn.size() != num_points*kn)
   {
      std::cout <<  "Err: points neighbors: nn.size() != num_points*kn" << std::endl;
      return false;
   }

   //debug (print neighbors)
   /*
   for (auto np = 0;np < num_points;np++)
   {
      std::cout <<  np << " points neighbors: ";
      for (auto k = 0; k < kn; k++)
      {
        std::cout << nn.at(np+k) << " ,";
      }
      std::cout << std::endl;  
   }
   */
   
   //std::vector<Eigen::Vector4d> normals;
   std::vector<Eigen::Matrix4d> covs;//not used
   std::unique_ptr<glim::CloudCovarianceEstimation> covariance_estimation;
   covariance_estimation.reset(new glim::CloudCovarianceEstimation(2));//num_threads
   covariance_estimation->estimate(temp, nn, normals, covs);
   
   if((normals.size() != points.size()) ||
        (covs.size() != points.size()))
   {
      std::cout <<  "Err: out arrs are not equals" << std::endl;
      return false;
   }

    //direction ambiguaty resolving
    /*
    https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html
    The solution to this problem is trivial if the viewpoint  is in fact known. To orient all normals  consistently
    towards the viewpoint, they need to satisfy the equation:
    points.at(index).dot(normals.at(index))<0
    */
   size_t index = 0;
  
   //spdlog::warn("Nota Bene: NO  direction ambiguaty resolving usage");
  
   for(index =0;index < normals.size();index++)
   {
        //Why ALL dot products in AVIA are negative???
        //std::cout << points.at(index).dot(normals.at(index)) << std::endl;
        if(temp.at(index).dot(normals.at(index))>=0)
        {
          //flip normal' vector
          normals.at(index).x() = normals.at(index).x() * (-1);
          normals.at(index).y() = normals.at(index).y() * (-1);
          normals.at(index).z() = normals.at(index).z() * (-1);
        }

        points.at(index).point_normal = normals.at(index);

   }
   
   //debug (print normals)
   /*
   index = 0;
   for(auto &&norm : normals)
   {
     std::cout <<  index++ << " point normal: ";
     //std::cout <<  norm << std::endl;
     std::cout << std::fixed << std::setprecision(6) << "x : " <<  norm.x() << " \t y : " <<  norm.y() << " \t z : " <<  norm.z() << std::endl;
   }
   */
 
  return true;
}
//normal calculation above



};//Reconstructor


//////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glim_rosbag input_rosbag_path" << std::endl;
    return 0;
  }

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto glim = std::make_shared<glim::GlimROS>(options);

  // List topics
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  const std::string imu_topic = config_ros.param<std::string>("glim_ros", "imu_topic", "/imu");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "/points");
  const std::string image_topic = config_ros.param<std::string>("glim_ros", "image_topic", "/image");
  std::vector<std::string> topics = {imu_topic, points_topic, image_topic};

  rosbag2_storage::StorageFilter filter;
  spdlog::info("topics:");
  for (const auto& topic : topics) {
    spdlog::info("- {}", topic);
    filter.topics.push_back(topic);
  }

  //
  std::unordered_map<std::string, std::vector<glim::GenericTopicSubscription::Ptr>> subscription_map;
  for (const auto& sub : glim->extension_subscriptions()) {
    spdlog::info("- {} (ext)", sub->topic);
    filter.topics.push_back(sub->topic);
    subscription_map[sub->topic].push_back(sub);
  }

  // List input rosbag filenames
  std::vector<std::string> bag_filenames;

  for (int i = 1; i < argc; i++) {
    std::vector<std::string> filenames;
    glob_t globbuf;
    int ret = glob(argv[i], 0, nullptr, &globbuf);
    for (int i = 0; i < globbuf.gl_pathc; i++) {
      filenames.push_back(globbuf.gl_pathv[i]);
    }
    globfree(&globbuf);

    bag_filenames.insert(bag_filenames.end(), filenames.begin(), filenames.end());
  }
  std::sort(bag_filenames.begin(), bag_filenames.end());

  spdlog::info("bag_filenames:");
  for (const auto& bag_filename : bag_filenames) {
    spdlog::info("- {}", bag_filename);
  }

  // Playback range settings
  double delay = 0.0;
  glim->declare_parameter<double>("delay", delay);
  glim->get_parameter<double>("delay", delay);

  double start_offset = 0.0;
  glim->declare_parameter<double>("start_offset", start_offset);
  glim->get_parameter<double>("start_offset", start_offset);

  double playback_duration = 0.0;
  glim->declare_parameter<double>("playback_duration", playback_duration);
  glim->get_parameter<double>("playback_duration", playback_duration);

  double playback_until = 0.0;
  glim->declare_parameter<double>("playback_until", playback_until);
  glim->get_parameter<double>("playback_until", playback_until);

  // Playback speed settings
  const double playback_speed = config_ros.param<double>("glim_ros", "playback_speed", 100.0);
  const auto real_t0 = std::chrono::high_resolution_clock::now();
  rcutils_time_point_value_t bag_t0 = 0;
  SpeedCounter speed_counter;

  double end_time = std::numeric_limits<double>::max();
  glim->declare_parameter<double>("end_time", end_time);
  glim->get_parameter<double>("end_time", end_time);

  if (delay > 0.0) {
    spdlog::info("delaying {} sec", delay);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000)));
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////
  bool reconstruct = false;
  //Declared in GlimROS ctor
  //glim->declare_parameter<bool>("reconstruct", reconstruct);
  glim->get_parameter<bool>("reconstruct", reconstruct);
  Reconstructor rc;
  if(reconstruct)
  {
    if(!rc.init())
    {
      spdlog::error("Reconstructor initialization error!");
      return false;
    }
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // Bag read function
  const auto read_bag = [&](const std::string& bag_filename) {
    spdlog::info("opening {}", bag_filename);
    rosbag2_storage::StorageOptions options;
    options.uri = bag_filename;

    rosbag2_cpp::ConverterOptions converter_options;

    // rosbag2_cpp::Reader reader;
    std::unique_ptr<rosbag2_cpp::reader_interfaces::BaseReaderInterface> reader_;
    reader_ = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
    reader_->open(options, converter_options);

    if (reader_->get_metadata().compression_format != "") {
      spdlog::info("compression detected (format={})", reader_->get_metadata().compression_format);
      spdlog::info("opening bag with SequentialCompressionReader");
      reader_ = std::make_unique<rosbag2_compression::SequentialCompressionReader>();
      reader_->open(options, converter_options);
    }

    auto& reader = *reader_;
    reader.set_filter(filter);

    const auto topics_and_types = reader.get_all_topics_and_types();
    std::unordered_map<std::string, std::string> topic_type_map;
    for (const auto& topic : topics_and_types) {
      topic_type_map[topic.name] = topic.type;
    }

    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> points_serialization;
    rclcpp::Serialization<sensor_msgs::msg::Image> image_serialization;
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_image_serialization;

    while (reader.has_next()) {
      if (!rclcpp::ok()) {
        return false;
      }

      const auto msg = reader.read_next();
      const std::string topic_type = topic_type_map[msg->topic_name];
      const rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);

      const auto msg_time = get_msg_recv_timestamp(*msg);
      if (bag_t0 == 0) {
        bag_t0 = msg_time;
      }
      
      //spdlog::debug("msg_time: {} ({} sec)", msg_time / 1e9, (msg_time - bag_t0) / 1e9);

      if (start_offset > 0.0 && msg_time - bag_t0 < start_offset * 1e9) {
        spdlog::debug("skipping msg for start_offset ({} < {})", (msg_time - bag_t0) / 1e9, start_offset);
        continue;
      }

      if (playback_until > 0.0 && msg_time / 1e9 > playback_until) {
        spdlog::info("reached playback_until ({} < {})", msg_time / 1e9, playback_until);
        return false;
      }

      if (playback_duration > 0.0 && (msg_time - bag_t0) / 1e9 > playback_duration) {
        spdlog::info("reached playback_duration ({} > {})", (msg_time - bag_t0) / 1e9, playback_duration);
        return false;
      }

      const auto bag_elapsed = std::chrono::nanoseconds(msg_time - bag_t0);
      while (playback_speed > 0.0 && (std::chrono::high_resolution_clock::now() - real_t0) * playback_speed < bag_elapsed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      if (msg->topic_name == imu_topic) {
        if(!reconstruct)
        {
          auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
          imu_serialization.deserialize_message(&serialized_msg, imu_msg.get());
          glim->imu_callback(imu_msg);
        }
        else//debug
        {
           //auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
           //imu_serialization.deserialize_message(&serialized_msg, imu_msg.get());
           //spdlog::trace("IMU: {}.{}", imu_msg->header.stamp.sec, imu_msg->header.stamp.nanosec);
        }

      } else if (msg->topic_name == points_topic) {
        auto points_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        points_serialization.deserialize_message(&serialized_msg, points_msg.get());
///////////////////////////////////////
        if(reconstruct)
        {
          //rc.points_callback(points_msg);
          print_points(points_msg);
          //spdlog::trace("POINTS: {}.{}", points_msg->header.stamp.sec, points_msg->header.stamp.nanosec);
        }
//////////////////////////////////////        
        else
        {
          const size_t workload = glim->points_callback(points_msg);
          if (points_msg->header.stamp.sec + points_msg->header.stamp.nanosec * 1e-9 > end_time) {
            spdlog::info("end_time reached");
            return false;
          }
          if (workload > 5) {
            // Odometry estimation is behind
            const size_t sleep_msec = (workload - 4) * 5;
            spdlog::debug("throttling: {} msec (workload={})", sleep_msec, workload);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
          }
        }
      } else if (msg->topic_name == image_topic && topic_type == "sensor_msgs/msg/Image") {
        if(!reconstruct)
        {
          auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
          image_serialization.deserialize_message(&serialized_msg, image_msg.get());
          glim->image_callback(image_msg);
        }
      } else if (msg->topic_name == image_topic && topic_type == "sensor_msgs/msg/CompressedImage") {
        if(!reconstruct)
        {
          auto compressed_image_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
          compressed_image_serialization.deserialize_message(&serialized_msg, compressed_image_msg.get());

          auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
          cv_bridge::toCvCopy(*compressed_image_msg, "bgr8")->toImageMsg(*image_msg);
          glim->image_callback(image_msg);
        }
      }

      auto found = subscription_map.find(msg->topic_name);
      if (found != subscription_map.end()) {
        for (const auto& sub : found->second) {
          sub->insert_message_instance(serialized_msg);
        }
      }

      glim->timer_callback();
      speed_counter.update(msg_time / 1e9);

      const auto t0 = std::chrono::high_resolution_clock::now();
      while (glim->needs_wait()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (std::chrono::high_resolution_clock::now() - t0 > std::chrono::seconds(1)) {
          spdlog::warn("throttling timeout (an extension module may be hanged)");
          break;
        }
      }
    }

    return true;
  };

  // Read all rosbags
  bool auto_quit = false;
  glim->declare_parameter<bool>("auto_quit", auto_quit);
  glim->get_parameter<bool>("auto_quit", auto_quit);

  std::string dump_path = "/tmp/dump";
  glim->declare_parameter<std::string>("dump_path", dump_path);
  glim->get_parameter<std::string>("dump_path", dump_path);

  for (const auto& bag_filename : bag_filenames) {
    if (!read_bag(bag_filename)) {
      auto_quit = true;
      break;
    }
  }
/////////////////////////////////////////  
  if(reconstruct)
  {
    rc.do_reconstruct();  
    return 0;
  }
/////////////////////////////////////////
  if (!auto_quit) {
    rclcpp::spin(glim);
  }

  glim->wait(auto_quit);
  glim->save(dump_path);

  return 0;
}