#ifndef CAMERA_TRANS_PLUGIN_HH_
#define CAMERA_TRANS_PLUGIN_HH_

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Name.hh>
#include <gz/math/Matrix4.hh>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace gazebo_sim_plugin
{
class CameraTransPlugin :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
    CameraTransPlugin();
    ~CameraTransPlugin() override;

    void Configure(const gz::sim::Entity &entity,
                  const std::shared_ptr<const sdf::Element> &sdf,
                  gz::sim::EntityComponentManager &ecm,
                  gz::sim::EventManager &eventMgr) override;

    void PreUpdate(const gz::sim::UpdateInfo &info,
                  gz::sim::EntityComponentManager &ecm) override;

    void randOffset(double pos_mag, double rot_mag);

private:
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr mat_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr mount_pose_pub_;  // New publisher for mount pose

    std::string pub_mat_topic_ = "/camera/transform_matrix";
    std::string pub_pose_topic_ = "/camera/pose";
    std::string pub_mount_pose_topic_ = "/mount/pose";  // New topic for mount pose
    
    gz::sim::Entity modelEntity_;
    gz::sim::Entity cameraEntity_;
    gz::sim::Entity mountEntity_;  // New entity for mount link
    gz::sim::Link cameraLink_;
    std::thread ros_spinner_thread_;

    bool pos_offset_set_ = false;
    bool rot_offset_set_ = false;
    double pos_mag_;
    double rot_mag_;
    gz::math::Vector3d pos_offset_;
    gz::math::Quaterniond rot_offset_;
};
}

#endif