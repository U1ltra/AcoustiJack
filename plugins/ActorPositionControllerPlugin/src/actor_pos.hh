/*
 * ActorPositionControllerPlugin.hh
    * This plugin will subscribe to a target's ROS 2 topic to receive Pose messages 
    * and move the actor to the target's Pose position.
*/

#ifndef ACTOR_POSITION_CONTROLLER_PLUGIN_HH_
#define ACTOR_POSITION_CONTROLLER_PLUGIN_HH_

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Pose.hh>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <random>

namespace gazebo_sim_plugin
{
class ActorPositionControllerPlugin :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
    ActorPositionControllerPlugin();
    ~ActorPositionControllerPlugin() override;

    void Configure(const gz::sim::Entity &entity,
                  const std::shared_ptr<const sdf::Element> &sdf,
                  gz::sim::EntityComponentManager &ecm,
                  gz::sim::EventManager &eventMgr) override;

    void PreUpdate(const gz::sim::UpdateInfo &info,
                  gz::sim::EntityComponentManager &ecm) override;

private:
    void TriggerCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void StopCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void AtkPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void VicPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg);

    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr atk_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr trigger_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr vic_pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pose_vel_pub_;
    
    std::string pub_pose_topic_ = "/actor_following/pose";
    std::string atk_pose_topic_ = "/actor_following/atk_pose";
    std::string vic_pose_topic_ = "/actor_walking/pose";
    std::string trigger_topic_ = "/start_movement/atker";
    std::string stop_topic_ = "/stop_movement/actor_following";
    std::string pub_pose_vel_topic_ = "/actor_following/pose_vel";
    
    gz::sim::Entity actorEntity_;
    gz::math::Pose3d atk_target_pose_;
    gz::math::Pose3d vic_position_;
    
    // Movement state
    gz::math::Vector3d current_velocity_ = gz::math::Vector3d::Zero;
    bool movement_enabled_ = false;
    bool has_new_target_ = false;
    
    // Movement parameters
    double max_velocity_ = 1.0;  // meters per second
    double max_acceleration_ = 0.5;  // meters per second^2
    double max_angular_acceleration_ = 1.0;  // radians per second^2
    double min_vic_atk_dist_ = 1.0;  // minimum distance to victim in meters
    int latency_threshold_ = 300; // milliseconds. camera fps = 30. 1/30 = 0.0333. 0.3 is about 9/10 frames.

    rclcpp::Time lastRecvAtkPose{0};
    std::chrono::steady_clock::duration lastPoseUpdate{0};
    rclcpp::Duration latency = rclcpp::Duration(0, 0);
    std::thread ros_spinner_thread_;

    double noise_std_dev_;
    bool enable_noise_;
    std::default_random_engine random_generator_;
    std::normal_distribution<double> noise_distribution_;
    gz::math::Vector3d ApplyGaussianNoise(const gz::math::Vector3d& position);
};
}

#endif