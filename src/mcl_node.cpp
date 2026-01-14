#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // Important for transforms
#include "visualization_msgs/msg/marker_array.hpp"
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Represents a single hypothesis of the robot's state.
 * * Used for Task A1 (Initialization).
 */
struct Particle {
  /// @brief X coordinate in the map frame (meters)
  double x;
  /// @brief Y coordinate in the map frame (meters)
  double y;
  /// @brief Orientation in the map frame (radians)
  double theta;
  /// @brief Probability weight of this particle (0.0 to 1.0)
  double weight;
};

/**
 * @brief Monte Carlo Localization (MCL) Node.
 * * This node implements a Particle Filter to estimate the robot's pose.
 * It handles initialization, motion updates (prediction), measurement updates
 * (correction), resampling, and pose estimation.
 */
class MCLNode : public rclcpp::Node {
public:
  /**
   * @brief Constructor for the MCL Node.
   * * Initializes ROS 2 parameters, subscribers, publishers, and the initial
   * particle set.
   */
  MCLNode() : Node("mcl_node") {

    this->declare_parameter("particle_count", 500);
    particle_count_ = this->get_parameter("particle_count").as_int();

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
  std::vector<Particle> particles_;
  int particle_count_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;

  rclcpp::Time last_time_;
  bool first_map_received_ = false;

  double noise_v_ = 0.1;
  double noise_w_ = 0.05;

  std::mt19937 rng_{std::random_device{}()};

  /**
   * @brief Internal struct to store Map Landmarks.
   */
  struct Landmark {
    int id;
    double x;
    double y;
  };

  std::vector<Landmark> map_landmarks_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  double measurement_noise_std_ = 2.5;

  /**
   * @brief Task A1: Particle Initialization.
   * * Uniformly distributes particles across the defined map bounds
   * (-10m to 10m) and assigns them equal initial weights.
   */
  void initialize_particles() {
    RCLCPP_INFO(this->get_logger(), "Initializing %d particles...",
                particle_count_);
    particles_.clear();

    double min_x = -10.0;
    double max_x = 10.0;
    double min_y = -10.0;
    double max_y = 10.0;

    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    double initial_weight = 1.0 / static_cast<double>(particle_count_);

    for (int i = 0; i < particle_count_; ++i) {
      Particle p;
      p.x = dist_x(rng_);
      p.y = dist_y(rng_);
      p.theta = dist_theta(rng_);
      p.weight = initial_weight;

      particles_.push_back(p);
    }

    RCLCPP_INFO(this->get_logger(),
                "Initialization complete. Publishing particles.");
    publish_particles();
  }

  /**
   * @brief Helper function to normalize angles.
   * * Ensures angles stay within [-PI, PI] to avoid wrapping errors.
   * @param angle The angle in radians.
   * @return Normalized angle in radians.
   */
  double normalize_angle(double angle) {
    while (angle > M_PI)
      angle -= 2.0 * M_PI;
    while (angle < -M_PI)
      angle += 2.0 * M_PI;
    return angle;
  }

  /**
   * @brief Task A2: Motion Update (Prediction Step).
   * * Applies the kinematic motion model to propagate particles based on
   * odometry. Adds Gaussian noise to simulate the uncertainty of robot motion.
   * * @param msg The noisy odometry message containing linear (v) and angular
   * (w) velocity.
   */
  void motion_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {

    rclcpp::Time current_time = msg->header.stamp;

    if (last_time_.nanoseconds() == 0) {
      last_time_ = current_time;
      return;
    }

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt < 0.001)
      return;

    double v = msg->twist.twist.linear.x;
    double w = msg->twist.twist.angular.z;

    std::normal_distribution<double> dist_v(0.0, noise_v_);
    std::normal_distribution<double> dist_w(0.0, noise_w_);

    for (auto &p : particles_) {
      double v_noisy = v + dist_v(rng_);
      double w_noisy = w + dist_w(rng_);

      p.x += v_noisy * dt * std::cos(p.theta);
      p.y += v_noisy * dt * std::sin(p.theta);
      p.theta += w_noisy * dt;

      p.theta = normalize_angle(p.theta);
    }

    publish_particles();
  }

  /**
   * @brief Callback to load the Ground Truth map.
   * * Parses the PointCloud2 message and stores landmarks in memory.
   * Run only once at startup.
   * * @param msg PointCloud2 message containing landmark locations.
   */
  void map_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!map_landmarks_.empty())
      return;

    int point_step = msg->point_step;
    for (size_t i = 0; i < msg->width; ++i) {
      uint8_t *ptr = &msg->data[i * point_step];

      float x = *reinterpret_cast<float *>(ptr);
      float y = *reinterpret_cast<float *>(ptr + 4);
      int id = *reinterpret_cast<int *>(ptr + 12);

      map_landmarks_.push_back(
          {id, static_cast<double>(x), static_cast<double>(y)});
    }
    RCLCPP_INFO(this->get_logger(), "Map loaded with %zu landmarks.",
                map_landmarks_.size());
  }

  /**
   * @brief Task A3: Measurement Update (Correction Step).
   * * Calculates the likelihood (weight) of each particle by comparing
   * observed landmarks against the known map.
   * * Triggers Resampling (Task A4) and Pose Estimation (Task A5).
   * * @param msg PointCloud2 message containing observed landmarks relative to
   * the robot.
   */
  void sensor_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (map_landmarks_.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Waiting for map...");
      return;
    }

    std::vector<Landmark> observations;
    int point_step = msg->point_step;
    for (size_t i = 0; i < msg->width; ++i) {
      uint8_t *ptr = &msg->data[i * point_step];
      float x = *reinterpret_cast<float *>(ptr);
      float y = *reinterpret_cast<float *>(ptr + 4);
      int id = *reinterpret_cast<int *>(ptr + 12);
      observations.push_back(
          {id, static_cast<double>(x), static_cast<double>(y)});
    }

    double total_weight = 0.0;

    for (auto &p : particles_) {
      double particle_likelihood = 1.0;

      for (const auto &obs : observations) {

        double cos_theta = std::cos(p.theta);
        double sin_theta = std::sin(p.theta);

        double predicted_x = p.x + (obs.x * cos_theta - obs.y * sin_theta);
        double predicted_y = p.y + (obs.x * sin_theta + obs.y * cos_theta);

        double dist_sq = 100.0;

        for (const auto &lm : map_landmarks_) {
          if (lm.id == obs.id) {
            double dx = predicted_x - lm.x;
            double dy = predicted_y - lm.y;
            dist_sq = dx * dx + dy * dy;
            break;
          }
        }

        double exponent =
            -dist_sq / (2.0 * measurement_noise_std_ * measurement_noise_std_);
        particle_likelihood *= std::exp(exponent);
      }

      p.weight = particle_likelihood;
      total_weight += p.weight;
    }

    if (total_weight > 0.0) {
      for (auto &p : particles_) {
        p.weight /= total_weight;
      }
    } else {

      RCLCPP_WARN(this->get_logger(), "All particles lost! Resetting weights.");
    initialize_particles();
    return;
    }

    resample_particles();

    estimate_pose();
    publish_particles();
  }

  /**
   * @brief Task A4: Low-Variance Resampling.
   * * Selects a new set of particles based on their weights.
   * Particles with higher weights are more likely to be duplicated,
   * while particles with lower weights are likely to be dropped.
   */
  void resample_particles() {

    std::vector<Particle> new_particles;
    new_particles.reserve(particle_count_);

    double r = 1.0 / static_cast<double>(particle_count_);

    std::uniform_real_distribution<double> dist(0.0, r);
    double U = dist(rng_);

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
   * * Computes the weighted mean of the particle cloud to estimate the robot's
   * pose. Uses vector math for orientation to avoid circular wrapping issues.
   * Publishes the result to /mcl_pose.
   */
  void estimate_pose() {
    double mean_x = 0.0;
    double mean_y = 0.0;
    double mean_cos = 0.0;
    double mean_sin = 0.0;

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
    pose_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, mean_theta);
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    pose_pub_->publish(pose_msg);
  }

  /**
   * @brief Task A6: Visualization.
   * * Converts particles into a PoseArray message for visualization in RViz.
   */
  void publish_particles() {
    auto msg = geometry_msgs::msg::PoseArray();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "map";

    for (const auto &p : particles_) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p.x;
      pose.position.y = p.y;
      pose.orientation.w = std::cos(p.theta / 2.0);
      pose.orientation.z = std::sin(p.theta / 2.0);
      msg.poses.push_back(pose);
    }
    particle_pub_->publish(msg);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MCLNode>());
  rclcpp::shutdown();
  return 0;
}