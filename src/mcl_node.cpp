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

struct Particle {
  double x;
  double y;
  double theta;
  double weight;
};

class MCLNode : public rclcpp::Node {
public:
  MCLNode() : Node("mcl_node") {
    this->declare_parameter("particle_count", 1000);
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

  double noise_v_ = 0.02;
  double noise_w_ = 0.01;

  std::mt19937 rng_{std::random_device{}()};

  struct Landmark {
    int id;
    double x;
    double y;
  };

  std::vector<Landmark> map_landmarks_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  double measurement_noise_std_ = 0.316;

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

  double normalize_angle(double angle) {
    while (angle > M_PI)
      angle -= 2.0 * M_PI;
    while (angle < -M_PI)
      angle += 2.0 * M_PI;
    return angle;
  }

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

    // FIXED: Get velocities in MAP frame (not robot frame)
    double vx = msg->twist.twist.linear.x;
    double vy = msg->twist.twist.linear.y;
    double w = msg->twist.twist.angular.z;

    std::normal_distribution<double> dist_vx(0.0, noise_v_);
    std::normal_distribution<double> dist_vy(0.0, noise_v_);
    std::normal_distribution<double> dist_w(0.0, noise_w_);

    for (auto &p : particles_) {
      double vx_noisy = vx + dist_vx(rng_);
      double vy_noisy = vy + dist_vy(rng_);
      double w_noisy = w + dist_w(rng_);

      // Update in map frame
      p.x += vx_noisy * dt;
      p.y += vy_noisy * dt;
      p.theta += w_noisy * dt;

      p.theta = normalize_angle(p.theta);
    }

    publish_particles();
  }

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

        double dist_sq = 10000.0;

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