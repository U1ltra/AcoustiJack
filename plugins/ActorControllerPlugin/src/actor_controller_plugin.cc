// TODO Path based updates
// TODO Animation sync

#include "actor_controller_plugin.hh"
#include <gz/plugin/Register.hh>

using namespace gazebo_sim_plugin;

ActorControllerPlugin::ActorControllerPlugin()
    : linear_velocity_(0, 0, 0), angular_velocity_z_(0),
      noise_std_dev_(0.0), enable_noise_(false),
      random_generator_(std::random_device{}())
{}

ActorControllerPlugin::~ActorControllerPlugin()
{
    rclcpp::shutdown();
    if (ros_spinner_thread_.joinable()) {
        ros_spinner_thread_.join();
    }
}

void ActorControllerPlugin::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &)
{
  // Save the actor entity reference
  this->actorEntity_ = entity;

  // Initialize ROS 2
  if (!rclcpp::ok())
    rclcpp::init(0, nullptr);

  this->ros_node_ = std::make_shared<rclcpp::Node>("actor_controller_plugin");

  // Get preset velocity parameters from SDF
  if (sdf->HasElement("velocity_x"))
    this->initial_velocity_.X(sdf->Get<double>("velocity_x"));
  if (sdf->HasElement("velocity_y"))
    this->initial_velocity_.Y(sdf->Get<double>("velocity_y"));
  if (sdf->HasElement("velocity_z"))
    this->initial_velocity_.Z(sdf->Get<double>("velocity_z"));
  if (sdf->HasElement("angular_velocity"))
    this->angular_velocity_z_ = sdf->Get<double>("angular_velocity");
  if (sdf->HasElement("trigger_topic"))
    trigger_topic_ = sdf->Get<std::string>("trigger_topic");
  // Publish the actor's pose on a topic given in the SDF (defaults to /actor_pose_default)
  if (sdf->HasElement("pub_pose_topic"))
        pub_pose_topic_ = sdf->Get<std::string>("pub_pose_topic");
  if (sdf->HasElement("use_social_force"))
    use_social_force_ = sdf->Get<bool>("use_social_force");
  
  // Gaussian noise parameters
  if (sdf->HasElement("noise_std_dev"))
    this->noise_std_dev_ = sdf->Get<double>("noise_std_dev");
  if (sdf->HasElement("enable_noise"))
    this->enable_noise_ = sdf->Get<bool>("enable_noise");
  
  linear_velocity_ = initial_velocity_;

  // Initialize Gaussian noise distribution
  noise_distribution_ = std::normal_distribution<double>(0.0, noise_std_dev_);

  // Subscribe to trigger topic
  this->trigger_sub_ = this->ros_node_->create_subscription<std_msgs::msg::Bool>(
    trigger_topic_, 10, 
    std::bind(&ActorControllerPlugin::TriggerCallback, this, std::placeholders::_1));
  this->social_force_sub_ = this->ros_node_->create_subscription<geometry_msgs::msg::TwistStamped>(
    social_force_topic_, 10, 
    std::bind(&ActorControllerPlugin::SocialForceCallback, this, std::placeholders::_1));
  this->pose_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::Pose>(
      pub_pose_topic_, 10);
  this->pose_vel_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      pub_pose_vel_topic_, 10);

  // Set custom animation time from this plugin
  auto animTimeComp = ecm.Component<gz::sim::components::AnimationTime>(entity);
  if (nullptr == animTimeComp)
  {
    ecm.CreateComponent(entity, gz::sim::components::AnimationTime());
  }

  // Start a separate thread to spin the ROS 2 node
  this->ros_spinner_thread_ = std::thread([this]() {
    rclcpp::spin(this->ros_node_);
  });

  RCLCPP_INFO(this->ros_node_->get_logger(), 
              "Actor Controller Plugin configured with noise: %s, std_dev: %f", 
              enable_noise_ ? "enabled" : "disabled", noise_std_dev_);
}

void ActorControllerPlugin::TriggerCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    movement_enabled_ = msg->data;
    RCLCPP_INFO(this->ros_node_->get_logger(), 
                "Movement %s", movement_enabled_ ? "enabled" : "disabled");
}

void ActorControllerPlugin::SocialForceCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    if (!use_social_force_)
        return;
    linear_velocity_.X(msg->twist.linear.x);
    linear_velocity_.Y(msg->twist.linear.y);
    linear_velocity_.Z(msg->twist.linear.z);
}

gz::math::Vector3d ActorControllerPlugin::ApplyGaussianNoise(const gz::math::Vector3d& position)
{
    if (!enable_noise_)
        return position;
    
    // Generate Gaussian noise for each axis
    double noise_x = noise_distribution_(random_generator_);
    double noise_y = noise_distribution_(random_generator_);
    double noise_z = noise_distribution_(random_generator_);
    
    return position + gz::math::Vector3d(noise_x, noise_y, noise_z);
}

void ActorControllerPlugin::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm)
{
  // Create a Model object using the actor entity
  gz::sim::Model model(this->actorEntity_);

  if (!this->use_social_force_) 
    this->linear_velocity_ = this->initial_velocity_;
  
  // Retrieve the current pose component
  auto poseComp = ecm.Component<gz::sim::components::Pose>(this->actorEntity_);
  if (poseComp)
  {
      // Get the current pose
      gz::math::Pose3d currentPose = poseComp->Data();
      
      // Publish the current pose
      geometry_msgs::msg::Pose pose_msg;
      // Convert Gazebo pose to ROS pose message
      pose_msg.position.x = currentPose.Pos().X();
      pose_msg.position.y = currentPose.Pos().Y();
      pose_msg.position.z = currentPose.Pos().Z();
      pose_msg.orientation.w = currentPose.Rot().W();
      pose_msg.orientation.x = currentPose.Rot().X();
      pose_msg.orientation.y = currentPose.Rot().Y();
      pose_msg.orientation.z = currentPose.Rot().Z();
      this->pose_pub_->publish(pose_msg);

      // Publish the current pose and velocity
      geometry_msgs::msg::TwistStamped pose_vel_msg;
      pose_vel_msg.header.stamp = this->ros_node_->now();
      pose_vel_msg.header.frame_id = "world";
      pose_vel_msg.twist.linear.x = linear_velocity_.X();
      pose_vel_msg.twist.linear.y = linear_velocity_.Y();
      pose_vel_msg.twist.linear.z = linear_velocity_.Z();
      pose_vel_msg.twist.angular.x = currentPose.Pos().X();
      pose_vel_msg.twist.angular.y = currentPose.Pos().Y();
      pose_vel_msg.twist.angular.z = currentPose.Pos().Z();
      this->pose_vel_pub_->publish(pose_vel_msg);
  }

  if (poseComp && movement_enabled_)
  {
    // Get the current pose
    gz::math::Pose3d currentPose = poseComp->Data();
    auto initialPose = currentPose;
    
    // Calculate the time step (dt) based on the simTime provided in UpdateInfo
    std::chrono::duration<double> dtDuration = info.simTime - this->lastUpdate;
    double dt = dtDuration.count();

    this->lastUpdate = info.simTime;

    // Update the position based on linear velocity (considering time step)
    gz::math::Vector3d newPosition = currentPose.Pos() + this->linear_velocity_ * dt;
    
    // Apply Gaussian noise to the position
    newPosition = ApplyGaussianNoise(newPosition);
    
    currentPose.Pos() = newPosition;

    // Update the rotation based on angular velocity (considering time step)
    // Use axis-angle for rotation around the Z-axis
    gz::math::Quaterniond deltaRotation(gz::math::Vector3d(0, 0, this->angular_velocity_z_ * dt));

    // Apply the updated rotation
    currentPose.Rot() *= deltaRotation;

    // Apply the updated pose
    double distanceTraveled = (currentPose.Pos() - initialPose.Pos()).Length();
    *poseComp = gz::sim::components::Pose(currentPose);

    // Mark as a one-time-change so that the change is propagated to the GUI
    ecm.SetChanged(this->actorEntity_,
        gz::sim::components::Pose::typeId, gz::sim::ComponentState::OneTimeChange);

    ///*

    // TODO Animation sync

    // Animation time component to be used for scaling animation speed by velocity
    auto animTimeComp = ecm.Component<gz::sim::components::AnimationTime>(this->actorEntity_);

    if (animTimeComp) {
      if (distanceTraveled <= 0.0) {
        distanceTraveled = std::numeric_limits<double>::min();
      }

      // Scale animation time inversely with distance traveled
      double timeScale = 1.0 / distanceTraveled;

      // Compute new animation time based on dt and time scale
      auto animTime = animTimeComp->Data() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt * timeScale));

      // Apply the updated animation time
      *animTimeComp = gz::sim::components::AnimationTime(animTime);

      // Mark as a one-time-change so that the change is propagated to the GUI
      ecm.SetChanged(this->actorEntity_,
          gz::sim::components::AnimationTime::typeId, gz::sim::ComponentState::OneTimeChange);
    }
  }

  this->lastUpdate = info.simTime; 
}

void ActorControllerPlugin::PostUpdate(
    const gz::sim::UpdateInfo &,
    const gz::sim::EntityComponentManager &)
{
  // Optionally read sensor data, log, or perform post-update tasks
  // (In this case, it's a placeholder as it's not used currently)
}



GZ_ADD_PLUGIN(
    gazebo_sim_plugin::ActorControllerPlugin,
    gz::sim::System,
    gazebo_sim_plugin::ActorControllerPlugin::ISystemConfigure,
    gazebo_sim_plugin::ActorControllerPlugin::ISystemPreUpdate,
    gazebo_sim_plugin::ActorControllerPlugin::ISystemPostUpdate
)
