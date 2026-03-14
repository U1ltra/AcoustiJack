// camera_pose_plugin.cc
#include "camera_trans.hh"
#include <gz/plugin/Register.hh>

using namespace gazebo_sim_plugin;

CameraTransPlugin::CameraTransPlugin()
{
}

CameraTransPlugin::~CameraTransPlugin()
{
    rclcpp::shutdown();
    if (this->ros_spinner_thread_.joinable())
    {
        this->ros_spinner_thread_.join();
    }
}

void CameraTransPlugin::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &)
{
    this->modelEntity_ = entity;
    
    // Initialize ROS 2
    if (!rclcpp::ok())
    rclcpp::init(0, nullptr);

    this->ros_node_ = std::make_shared<rclcpp::Node>("camera_translation_plugin");
    gz::sim::Model model(this->modelEntity_);
    
    // Find the camera link entity
    this->cameraEntity_ = model.LinkByName(ecm, "camera_link");
    
    // Find the mount link entity
    this->mountEntity_ = model.LinkByName(ecm, "cgo3_mount_link");
    
    // Parse SDF parameters
    if (sdf->HasElement("pub_mat_topic"))
        this->pub_mat_topic_ = sdf->Get<std::string>("pub_mat_topic");
    if (sdf->HasElement("pub_pose_topic"))
        this->pub_pose_topic_ = sdf->Get<std::string>("pub_pose_topic");
    if (sdf->HasElement("pub_mount_pose_topic"))
        this->pub_mount_pose_topic_ = sdf->Get<std::string>("pub_mount_pose_topic");
    if (sdf->HasElement("pos_mag"))
    {
        pos_offset_set_ = true;
        pos_mag_ = sdf->Get<double>("pos_mag");
    }
    if (sdf->HasElement("rot_mag"))
    {
        rot_offset_set_ = true;
        rot_mag_ = sdf->Get<double>("rot_mag");
    }

    this->mat_pub_ = this->ros_node_->create_publisher<std_msgs::msg::Float64MultiArray>(
        this->pub_mat_topic_, 10);
    this->pose_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::Pose>(
        this->pub_pose_topic_, 10);
    this->mount_pose_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::Pose>(
        this->pub_mount_pose_topic_, 10);

    // print offset
    RCLCPP_INFO(this->ros_node_->get_logger(), "Position offset: %f %f %f", 
                pos_offset_.X(), pos_offset_.Y(), pos_offset_.Z());
    gz::math::Vector3d euler = rot_offset_.Euler();
    RCLCPP_INFO(this->ros_node_->get_logger(), "Orientation offset: %f %f %f", 
                euler.X(), euler.Y(), euler.Z());
}

void CameraTransPlugin::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm)
{
    if (this->cameraEntity_ == gz::sim::kNullEntity)
    {
        RCLCPP_WARN_THROTTLE(this->ros_node_->get_logger(), 
                             *this->ros_node_->get_clock(), 5000,
                             "Camera link not found");
        return;
    }

    // Get and publish camera pose
    gz::math::Pose3d cameraWorldPose = gz::sim::worldPose(this->cameraEntity_, ecm);
    
    geometry_msgs::msg::Pose camera_pose_msg;
    camera_pose_msg.position.x = cameraWorldPose.Pos().X();
    camera_pose_msg.position.y = cameraWorldPose.Pos().Y();
    camera_pose_msg.position.z = cameraWorldPose.Pos().Z();
    camera_pose_msg.orientation.w = cameraWorldPose.Rot().W();
    camera_pose_msg.orientation.x = cameraWorldPose.Rot().X();
    camera_pose_msg.orientation.y = cameraWorldPose.Rot().Y();
    camera_pose_msg.orientation.z = cameraWorldPose.Rot().Z();
    this->pose_pub_->publish(camera_pose_msg);

    // Get and publish mount pose
    if (this->mountEntity_ != gz::sim::kNullEntity)
    {
        gz::math::Pose3d mountWorldPose = gz::sim::worldPose(this->mountEntity_, ecm);
        
        geometry_msgs::msg::Pose mount_pose_msg;
        mount_pose_msg.position.x = mountWorldPose.Pos().X();
        mount_pose_msg.position.y = mountWorldPose.Pos().Y();
        mount_pose_msg.position.z = mountWorldPose.Pos().Z();
        mount_pose_msg.orientation.w = mountWorldPose.Rot().W();
        mount_pose_msg.orientation.x = mountWorldPose.Rot().X();
        mount_pose_msg.orientation.y = mountWorldPose.Rot().Y();
        mount_pose_msg.orientation.z = mountWorldPose.Rot().Z();
        this->mount_pose_pub_->publish(mount_pose_msg);
    }
    else
    {
        RCLCPP_WARN_THROTTLE(this->ros_node_->get_logger(), 
                             *this->ros_node_->get_clock(), 5000,
                             "Mount link not found");
    }

    // Generate transformation matrices (existing code)
    std::vector<gz::math::Matrix4d> transforms;
    std::vector<gz::math::Matrix4d> inv_transforms;
    
    // Add the initial pose transformation
    gz::math::Pose3d currentPose = cameraWorldPose;
    transforms.push_back(gz::math::Matrix4d(currentPose));
    inv_transforms.push_back(transforms.back().Inverse());
    
    // Generate 3 additional random transformations
    for (int i = 0; i < 3; ++i) {
        // Generate random offset
        randOffset(pos_mag_, rot_mag_);
        
        // Apply offsets if enabled
        if (this->pos_offset_set_) {
            currentPose.Pos() += pos_offset_;
        }
        if (this->rot_offset_set_) {
            currentPose.Rot() = currentPose.Rot() * rot_offset_;
        }
        
        // Create and store transformation matrices
        transforms.push_back(gz::math::Matrix4d(currentPose));
        inv_transforms.push_back(transforms.back().Inverse());
    }
    
    // Build the message data array
    std_msgs::msg::Float64MultiArray mat_msg;
    mat_msg.data.reserve(128); // Pre-allocate space for all elements
    
    // Add all transformation matrices to the message
    for (size_t i = 0; i < transforms.size(); ++i) {
        // Add forward transform
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                mat_msg.data.push_back(transforms[i](row, col));
            }
        }
        
        // Add inverse transform
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                mat_msg.data.push_back(inv_transforms[i](row, col));
            }
        }
    }

    this->mat_pub_->publish(mat_msg);
    pos_offset_.Set(0, 0, 0);
    rot_offset_ = gz::math::Quaterniond::Identity;
}

void CameraTransPlugin::randOffset(double pos_mag, double rot_mag)
{
    const double orientationNoiseRadians = rot_mag * M_PI / 180.0;

    // Generate random direction for position noise (unit vector)
    gz::math::Vector3d positionNoiseDirection;
    // Generate random components between -1 and 1 for each axis
    pos_offset_.Set(
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0,
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0,
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0
    );
    // Normalize and scale to desired magnitude
    pos_offset_ = pos_offset_.Normalize() * pos_mag;
    
    // Generate random axis for rotation noise (unit vector)
    gz::math::Vector3d rotationAxis;
    // Generate random components between -1 and 1 for each axis
    rotationAxis.Set(
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0,
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0,
        (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0
    );
    rotationAxis = rotationAxis.Normalize();
    rot_offset_ = gz::math::Quaterniond(rotationAxis, orientationNoiseRadians);
}

GZ_ADD_PLUGIN(
    gazebo_sim_plugin::CameraTransPlugin,
    gz::sim::System,
    gazebo_sim_plugin::CameraTransPlugin::ISystemConfigure,
    gazebo_sim_plugin::CameraTransPlugin::ISystemPreUpdate
)