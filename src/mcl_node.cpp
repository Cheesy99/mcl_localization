#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Represents a single hypothesis of the robot's state.
 * * Contains the 2D pose (x, y, theta) and the associated probability weight.
 */
struct Particle {
  double x;      /**< X coordinate in map frame (meters) */
  double y;      /**< Y coordinate in map frame (meters) */
  double theta;  /**< Heading angle in radians [-PI, PI] */
  double weight; /**< Importance weight [0.0, 1.0] */
};

/**
 * @class MCLNode
 * @brief ROS 2 Node implementing Monte Carlo Localization (Particle Filter).
 * * This node estimates a robot's pose by maintaining a set of particles that 
 * evolve through motion prediction and sensor-based correction.
 */
class MCLNode : public rclcpp::Node {
public:
  /**
   * @brief Constructor for the MCL Node.
   * * Initializes parameters, sets up ROS 2 subscribers/publishers, and 
   * performs the initial uniform distribution of particles.
   */
  MCLNode() : Node("mcl_node") {
    this->declare_parameter("particle_count", 1000);
    this->declare_parameter("noise_v", 0.05);
    this->declare_parameter("noise_w", 0.02);
    this->declare_parameter("sensor_std", 0.316);

    particle_count_ = this->get_parameter("particle_count").as_int();
    noise_v_ = this->get_parameter("noise_v").as_double();
    noise_w_ = this->get_parameter("noise_w").as_double();
    measurement_noise_std_ = this->get_parameter("sensor_std").as_double();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/robot_noisy", 10,
        std::bind(&MCLNode::motion_callback, this, std::placeholders::_1));
    sensor_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/landmarks_observed", 10,
        std::bind(&MCLNode::sensor_callback, this, std::placeholders::_1));
    map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/landmarks_gt", 10,
        std::bind(&MCLNode::map_callback, this, std::placeholders::_1));

    pose_pub_ =
        this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/mcl_pose", 10);
    particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/mcl_particles", 10);

    initialize_particles();
  }

private:
  // --- ROS 2 Member Variables ---
  std::vector<Particle> particles_;
  int particle_count_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;

  // --- Filter State Variables ---
  rclcpp::Time last_time_;
  bool first_map_received_ = false;
  double noise_v_ = 0.02; /**< Standard deviation of linear velocity noise */
  double noise_w_ = 0.01; /**< Standard deviation of angular velocity noise */
  std::mt19937 rng_{std::random_device{}()};

  /** @brief Internal representation of a map landmark */
  struct Landmark {
    int id;
    double x;
    double y;
  };

  std::vector<Landmark> map_landmarks_;
  double measurement_noise_std_ = 0.316; /**< Std dev for Gaussian likelihood model */

  /**
   * @brief Task A1: Initialization.
   * * Spreads particles uniformly across a 20x20m area and assigns 
   * equal importance weights.
   */
  void initialize_particles() {
    RCLCPP_INFO(this->get_logger(), "Initializing %d particles...", particle_count_);
    particles_.clear();

    double min_x = -10.0, max_x = 10.0;
    double min_y = -10.0, max_y = 10.0;

    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    double initial_weight = 1.0 / static_cast<double>(particle_count_);

    for (int i = 0; i < particle_count_; ++i) {
      particles_.push_back({dist_x(rng_), dist_y(rng_), dist_theta(rng_), initial_weight});
    }

    RCLCPP_INFO(this->get_logger(), "Initialization complete.");
    publish_particles();
  }

  /**
   * @brief Helper to wrap angles into the [-PI, PI] range.
   * @param angle Input angle in radians.
   * @return Normalized angle in radians.
   */
  double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  /**
   * @brief Task A2: Motion Update (Prediction).
   * * Propagates particles forward in time using noisy odometry data.
   * Uses a direct map-frame update based on the velocity components.
   * @param msg Odometry message containing current twist.
   */
  void motion_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    rclcpp::Time current_time = msg->header.stamp;
    if (last_time_.nanoseconds() == 0) {
      last_time_ = current_time;
      return;
    }

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;
    if (dt < 0.001) return;

    double vx = msg->twist.twist.linear.x;
    double vy = msg->twist.twist.linear.y;
    double w = msg->twist.twist.angular.z;

    std::normal_distribution<double> dist_vx(0.0, noise_v_);
    std::normal_distribution<double> dist_vy(0.0, noise_v_);
    std::normal_distribution<double> dist_w(0.0, noise_w_);

    for (auto &p : particles_) {
      p.x += (vx + dist_vx(rng_)) * dt;
      p.y += (vy + dist_vy(rng_)) * dt;
      p.theta = normalize_angle(p.theta + (w + dist_w(rng_)) * dt);
    }
    publish_particles();
  }

  /**
   * @brief Callback to load the Global Landmark Map.
   * * Stores the ground truth landmark positions. Only runs once.
   * @param msg PointCloud2 containing ground truth landmark data.
   */
  void map_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!map_landmarks_.empty()) return;

    int point_step = msg->point_step;
    for (size_t i = 0; i < msg->width; ++i) {
      uint8_t *ptr = &msg->data[i * point_step];
      float x = *reinterpret_cast<float *>(ptr);
      float y = *reinterpret_cast<float *>(ptr + 4);
      int id = *reinterpret_cast<int *>(ptr + 12);
      map_landmarks_.push_back({id, static_cast<double>(x), static_cast<double>(y)});
    }
    RCLCPP_INFO(this->get_logger(), "Map loaded with %zu landmarks.", map_landmarks_.size());
  }

  /**
   * @brief Task A3: Measurement Update (Correction).
   * * Updates particle weights based on the likelihood of observing 
   * landmarks from each particle's pose.
   * @param msg PointCloud2 containing observed landmarks from robot's sensor.
   */
  void sensor_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (map_landmarks_.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for map...");
      return;
    }

    // Parse sensor observations
    std::vector<Landmark> observations;
    int point_step = msg->point_step;
    for (size_t i = 0; i < msg->width; ++i) {
      uint8_t *ptr = &msg->data[i * point_step];
      float x = *reinterpret_cast<float *>(ptr);
      float y = *reinterpret_cast<float *>(ptr + 4);
      int id = *reinterpret_cast<int *>(ptr + 12);
      observations.push_back({id, static_cast<double>(x), static_cast<double>(y)});
    }

    double total_weight = 0.0;
    for (auto &p : particles_) {
      double particle_likelihood = 1.0;

      for (const auto &obs : observations) {
        // Transform local observation to global map coordinates using particle pose
        double cos_theta = std::cos(p.theta);
        double sin_theta = std::sin(p.theta);
        double predicted_x = p.x + (obs.x * cos_theta - obs.y * sin_theta);
        double predicted_y = p.y + (obs.x * sin_theta + obs.y * cos_theta);

        double dist_sq = 10000.0; // Large penalty for no match

        // Find matching landmark in the map
        for (const auto &lm : map_landmarks_) {
          if (lm.id == obs.id) {
            double dx = predicted_x - lm.x;
            double dy = predicted_y - lm.y;
            dist_sq = dx * dx + dy * dy;
            break;
          }
        }

        // Gaussian likelihood model
        double exponent = -dist_sq / (2.0 * measurement_noise_std_ * measurement_noise_std_);
        particle_likelihood *= std::exp(exponent);
      }
      p.weight = particle_likelihood;
      total_weight += p.weight;
    }

    // Normalize weights or reset if filter diverged
    if (total_weight > 0.0) {
      for (auto &p : particles_) p.weight /= total_weight;
    } else {
      RCLCPP_WARN(this->get_logger(), "All particles lost! Resetting weights.");
      initialize_particles();
      return;
    }

    resample_particles(); /**< Task A4 */
    estimate_pose();      /**< Task A5 */
    publish_particles();  /**< Task A6 */
  }

  /**
   * @brief Task A4: Low-Variance Resampling.
   * * Stochastic universal sampling to replace low-weight particles with 
   * high-weight ones while maintaining diversity.
   */
  void resample_particles() {
    std::vector<Particle> new_particles;
    new_particles.reserve(particle_count_);

    double r = 1.0 / static_cast<double>(particle_count_);
    double U = std::uniform_real_distribution<double>(0.0, r)(rng_);
    double c = particles_[0].weight;
    int i = 0;

    for (int m = 0; m < particle_count_; ++m) {
      double U_m = U + (m * r);
      while (U_m > c && i < particle_count_ - 1) {
        i++;
        c += particles_[i].weight;
      }
      Particle p = particles_[i];
      p.weight = 1.0 / particle_count_;
      new_particles.push_back(p);
    }
    particles_ = new_particles;
  }

  /**
   * @brief Task A5: Pose Estimation.
   * * Calculates the weighted mean of the particle set to determine 
   * the robot's most likely pose.
   */
  void estimate_pose() {
    double mean_x = 0.0, mean_y = 0.0;
    double mean_cos = 0.0, mean_sin = 0.0;

    for (const auto &p : particles_) {
      mean_x += p.x * p.weight;
      mean_y += p.y * p.weight;
      mean_cos += std::cos(p.theta) * p.weight;
      mean_sin += std::sin(p.theta) * p.weight;
    }

    double mean_theta = std::atan2(mean_sin, mean_cos);

    auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = mean_x;
    pose_msg.pose.pose.position.y = mean_y;

    tf2::Quaternion q;
    q.setRPY(0, 0, mean_theta);
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    pose_pub_->publish(pose_msg);
  }

  /**
   * @brief Task A6: Visualization.
   * * Converts the internal particle cloud into a PoseArray message 
   * for visualization in RViz.
   */
  void publish_particles() {
    auto msg = geometry_msgs::msg::PoseArray();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "map";

    for (const auto &p : particles_) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p.x;
      pose.position.y = p.y;
      tf2::Quaternion q;
      q.setRPY(0, 0, p.theta);
      pose.orientation = tf2::toMsg(q);
      msg.poses.push_back(pose);
    }
    particle_pub_->publish(msg);
  }
};

/**
 * @brief Entry point for the MCL node.
 */
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MCLNode>());
  rclcpp::shutdown();
  return 0;
}