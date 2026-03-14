#ifndef ACTOR_CONTROLLER_PLUGIN_HH_
#define ACTOR_CONTROLLER_PLUGIN_HH_

#include <gz/sim/System.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Actor.hh>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <gz/math/Vector3.hh>
#include <gz/math/SphericalCoordinates.hh>
#include <thread>
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/follow_me/follow_me.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/param/param.h>

#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <std_msgs/msg/bool.hpp> 
#include <std_msgs/msg/float64_multi_array.hpp>
#include <random>

// Define constants for world reference coordinates
constexpr double WORLD_REF_LAT = 47.397971057728974;
constexpr double WORLD_REF_LON = 8.546163739800146;
constexpr double WORLD_REF_ELEVATION = 0.0;
constexpr double WORLD_REF_HEADING = 0.0;

namespace gazebo_sim_plugin
{
class ActorControllerPlugin :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate,
    public gz::sim::ISystemPostUpdate
{
public:
    ActorControllerPlugin();
    ~ActorControllerPlugin() override;

    void Configure(const gz::sim::Entity &entity,
                  const std::shared_ptr<const sdf::Element> &sdf,
                  gz::sim::EntityComponentManager &ecm,
                  gz::sim::EventManager &eventMgr) override;

    void PreUpdate(const gz::sim::UpdateInfo &info,
                  gz::sim::EntityComponentManager &ecm) override;

    void PostUpdate(const gz::sim::UpdateInfo &info,
                   const gz::sim::EntityComponentManager &ecm) override;

private:
    void TriggerCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void TwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void SocialForceCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

    // ROS 2 components
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr trigger_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr social_force_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pose_vel_pub_; // vx, vy, vz, x, y, z (pose instead of angular velocity)
    std::string trigger_topic_ = "/start_movement";
    std::string twist_topic_ = "/actor_walking/twist";
    std::string social_force_topic_ = "/actor_walking/social_force";
    std::string pub_pose_topic_ = "/actor_walking/pose";
    std::string pub_pose_vel_topic_ = "/actor_walking/pose_vel";

    // Actor components
    gz::sim::Entity actorEntity_;
    gz::math::Vector3d linear_velocity_;
    gz::math::Vector3d initial_velocity_;
    double angular_velocity_z_;
    std::chrono::steady_clock::duration lastUpdate{0};
    std::thread ros_spinner_thread_;
    bool movement_enabled_ = false;
    bool use_social_force_ = false;

    double noise_std_dev_;
    bool enable_noise_;
    std::default_random_engine random_generator_;
    std::normal_distribution<double> noise_distribution_;
    gz::math::Vector3d ApplyGaussianNoise(const gz::math::Vector3d& position);
};
}
#endif // ACTOR_CONTROLLER_PLUGIN_HH_