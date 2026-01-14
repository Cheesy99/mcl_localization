#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <vector>
#include <random>
#include <cmath>

// Struct to hold individual particle data (Task A1)
struct Particle {
    double x;
    double y;
    double theta;
    double weight;
};

class MCLNode : public rclcpp::Node {
public:
    MCLNode() : Node("mcl_node") {
        // --- Parameters ---
        this->declare_parameter("particle_count", 100);
        particle_count_ = this->get_parameter("particle_count").as_int();

        // --- Subscribers ---
        // Task A2: Listen to noisy odometry for Motion Update
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/robot_noisy", 10, std::bind(&MCLNode::motion_callback, this, std::placeholders::_1));

        // Task A3: Listen to landmarks for Measurement Update
        // Note: Your fake_robot publishes 'landmarks_observed', acting as the sensor data
        sensor_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/landmarks_observed", 10, std::bind(&MCLNode::sensor_callback, this, std::placeholders::_1));

        // --- Publishers ---
        // Task A5: Publish the estimated pose (The "Result")
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/mcl_pose", 10);
        
        // Task A6: Publish particles for visualization in RViz
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/mcl_particles", 10);

        // --- Initialization ---
        initialize_particles();
    }

private:
    std::vector<Particle> particles_;
    int particle_count_;
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;

    // Random number generation
    std::mt19937 rng_{std::random_device{}()};


    // --- Task A1: Particle Initialization ---
    void initialize_particles() {
        RCLCPP_INFO(this->get_logger(), "Initializing %d particles...", particle_count_);
        particles_.clear();
        
        // TODO: Define map bounds (e.g., -10 to 10 meters)
        std::uniform_real_distribution<double> dist_x(-10.0, 10.0);
        std::uniform_real_distribution<double> dist_y(-10.0, 10.0);
        std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

        for (int i = 0; i < particle_count_; ++i) {
            Particle p;
            p.x = dist_x(rng_);
            p.y = dist_y(rng_);
            p.theta = dist_theta(rng_);
            p.weight = 1.0 / particle_count_; // Equal weights initially
            particles_.push_back(p);
        }
        publish_particles(); // Visualize initial state
    }

    // --- Task A2: Motion Update ---
    void motion_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // TODO: Extract velocity (v, w) from msg->twist.twist
        // TODO: Loop through particles and apply motion model:
        //       x_new = x + (v * cos(theta) * dt) + noise
        //       y_new = y + (v * sin(theta) * dt) + noise
        //       theta_new = theta + (w * dt) + noise
        
        // For now, we just log that we received data
        // RCLCPP_INFO(this->get_logger(), "Motion update received");
    }

    // --- Task A3 & A4: Measurement Update & Resampling ---
    void sensor_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 1. Task A3: Compute Likelihood (Weights)
        // TODO: Loop through particles
        //       For each particle, compare 'msg' (observation) vs 'map' (ground truth)
        //       Update p.weight
        
        // 2. Task A3: Normalize Weights
        // TODO: Sum all weights, divide each p.weight by sum

        // 3. Task A4: Resampling (Low Variance)
        // TODO: Create new set of particles based on weights
        resample_particles();

        // 4. Task A5: Estimate Pose
        estimate_pose();

        // 5. Task A6: Visualize
        publish_particles();
    }

    void resample_particles() {
        // TODO: Implement Low-Variance Resampling
    }

    // --- Task A5: Pose Estimation ---
    void estimate_pose() {
        double mean_x = 0, mean_y = 0, mean_theta_x = 0, mean_theta_y = 0;

        // TODO: Calculate weighted mean
        
        // Publish the result
        auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
        pose_msg.header.stamp = this->get_clock()->now();
        pose_msg.header.frame_id = "map";
        // pose_msg.pose.pose.position.x = mean_x; ...
        pose_pub_->publish(pose_msg);
    }

    // --- Task A6: Visualization ---
    void publish_particles() {
        auto msg = geometry_msgs::msg::PoseArray();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";

        for (const auto& p : particles_) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = p.x;
            pose.position.y = p.y;
            // Convert theta to quaternion for visualization
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