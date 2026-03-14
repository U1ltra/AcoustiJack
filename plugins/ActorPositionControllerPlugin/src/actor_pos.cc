#include "actor_pos.hh"
#include <gz/plugin/Register.hh>

using namespace gazebo_sim_plugin; // needed to find the class definition

ActorPositionControllerPlugin::ActorPositionControllerPlugin()
    : noise_std_dev_(0.0), enable_noise_(false),
      random_generator_(std::random_device{}())
{
}

ActorPositionControllerPlugin::~ActorPositionControllerPlugin()
{
    rclcpp::shutdown();
    if (this->ros_spinner_thread_.joinable())
    {
        this->ros_spinner_thread_.join();
    }
}

void ActorPositionControllerPlugin::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &)
{
    this->actorEntity_ = entity;

    if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

    this->ros_node_ = std::make_shared<rclcpp::Node>("actor_position_controller_plugin");

    // Required parameters
    if (sdf->HasElement("move_speed"))
        max_velocity_ = sdf->Get<double>("move_speed");
    if (sdf->HasElement("max_acceleration"))
        max_acceleration_ = sdf->Get<double>("max_acceleration");
    if (sdf->HasElement("vic_atk_dist"))
        min_vic_atk_dist_ = sdf->Get<double>("vic_atk_dist");
    if (sdf->HasElement("latency_threshold"))
        latency_threshold_ = sdf->Get<int>("latency_threshold");
    
    if (sdf->HasElement("pub_pose_topic"))
        pub_pose_topic_ = sdf->Get<std::string>("pub_pose_topic");
    // Get parameters from SDF if provided
    if (sdf->HasElement("atk_pose_topic"))
        atk_pose_topic_ = sdf->Get<std::string>("atk_pose_topic");
    if (sdf->HasElement("vic_pose_topic"))
        vic_pose_topic_ = sdf->Get<std::string>("vic_pose_topic");
    if (sdf->HasElement("trigger_topic"))
        trigger_topic_ = sdf->Get<std::string>("trigger_topic");
    if (sdf->HasElement("pub_pose_vel_topic"))
        pub_pose_vel_topic_ = sdf->Get<std::string>("pub_pose_vel_topic");

    // Gaussian noise parameters
    if (sdf->HasElement("noise_std_dev"))
        this->noise_std_dev_ = sdf->Get<double>("noise_std_dev");
    if (sdf->HasElement("enable_noise"))
        this->enable_noise_ = sdf->Get<bool>("enable_noise");

    // Initialize Gaussian noise distribution
    noise_distribution_ = std::normal_distribution<double>(0.0, noise_std_dev_);

    this->atk_pose_sub_ = this->ros_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        atk_pose_topic_, 10, 
        std::bind(&ActorPositionControllerPlugin::AtkPoseCallback, this, std::placeholders::_1));
    this->vic_pose_sub_ = this->ros_node_->create_subscription<geometry_msgs::msg::Pose>(
        vic_pose_topic_, 10, 
        std::bind(&ActorPositionControllerPlugin::VicPoseCallback, this, std::placeholders::_1));
    this->trigger_sub_ = this->ros_node_->create_subscription<std_msgs::msg::Bool>(
        trigger_topic_, 10, 
        std::bind(&ActorPositionControllerPlugin::TriggerCallback, this, std::placeholders::_1));
    this->stop_sub_ = this->ros_node_->create_subscription<std_msgs::msg::Bool>(
        stop_topic_, 10, 
        std::bind(&ActorPositionControllerPlugin::StopCallback, this, std::placeholders::_1));
    this->pose_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::Pose>(
        pub_pose_topic_, 10);
    this->pose_vel_pub_ = this->ros_node_->create_publisher<geometry_msgs::msg::TwistStamped>(
        pub_pose_vel_topic_, 10);

    this->lastRecvAtkPose = this->ros_node_->now();

    this->ros_spinner_thread_ = std::thread([this]() {
        rclcpp::spin(this->ros_node_);
    });

    RCLCPP_INFO(this->ros_node_->get_logger(), 
                "Actor Position Controller Plugin configured with noise: %s, std_dev: %f", 
                enable_noise_ ? "enabled" : "disabled", noise_std_dev_);
}

void ActorPositionControllerPlugin::TriggerCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    movement_enabled_ = msg->data;
}
void ActorPositionControllerPlugin::StopCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    movement_enabled_ = false;
    current_velocity_ = gz::math::Vector3d::Zero;
}

gz::math::Vector3d ActorPositionControllerPlugin::ApplyGaussianNoise(const gz::math::Vector3d& position)
{
    if (!enable_noise_)
        return position;
    
    // Generate Gaussian noise for each axis
    double noise_x = noise_distribution_(random_generator_);
    double noise_y = noise_distribution_(random_generator_);
    double noise_z = noise_distribution_(random_generator_);
    
    return position + gz::math::Vector3d(noise_x, noise_y, noise_z);
}

/*
 * Callback for receiving the target pose of the actor to follow.
 *
 * Regardless of the attack interval, the optimization script takes T(comp) + T(comm)
 * to send the next target pose using this callback. In our simulation, T(comm) is
 * negligible, so the latency is mostly due to computation time T(comp). And T(comp)
 * is mostly due to the optimization algorithm. So, the latency is mostly due to the
 * iteration number of the optimization algorithm.
 * 
 * Empirically, the latency is about 0.3s, 0.4s, and 0.5s for 3, 5, and 7 iterations respectively.
 */
void ActorPositionControllerPlugin::AtkPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{ 
    // atk_target_pose_ = vic_position_;

    rclcpp::Time now = this->ros_node_->now();
    rclcpp::Time msg_time = rclcpp::Time(msg->header.stamp);
    rclcpp::Duration commLatency = now - msg_time;

    // Store the world position target
    atk_target_pose_ = gz::math::Pose3d(
        msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
        msg->pose.orientation.w, msg->pose.orientation.x, 
        msg->pose.orientation.y, msg->pose.orientation.z);
    
    latency = now - this->lastRecvAtkPose;
    RCLCPP_INFO(this->ros_node_->get_logger(), "Received target pose at %f, %f",
        atk_target_pose_.Pos().X(), atk_target_pose_.Pos().Y());
    RCLCPP_INFO(this->ros_node_->get_logger(), "Communication latency: %f", 
        commLatency.seconds());
    RCLCPP_INFO(this->ros_node_->get_logger(), "Latency: %f", latency.seconds());
    this->lastRecvAtkPose = now;
}

void ActorPositionControllerPlugin::VicPoseCallback(
    const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Convert ROS pose to Gazebo pose
    vic_position_ = gz::math::Pose3d(
        msg->position.x, msg->position.y, msg->position.z,
        msg->orientation.w, msg->orientation.x, 
        msg->orientation.y, msg->orientation.z);

    // Only for moving the actor closer to the victim without optimized trajectory - TODO //
    // atk_target_pose_ = gz::math::Pose3d(
    //     msg->position.x, msg->position.y, msg->position.z,
    //     msg->orientation.w, msg->orientation.x,
    //     msg->orientation.y, msg->orientation.z);
}

/*
 * This function is called every simulation step.
 * Simulation step width is configured in the world file .sdf.
 * We are using a fixed step width of 0.004s.
 */
void ActorPositionControllerPlugin::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm)
{
    auto poseComp = ecm.Component<gz::sim::components::Pose>(this->actorEntity_);
    if (!poseComp)
        return;

    gz::math::Pose3d currentPose = poseComp->Data();

    // Publish the current pose
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = currentPose.Pos().X();
    pose_msg.position.y = currentPose.Pos().Y();
    pose_msg.position.z = currentPose.Pos().Z();
    pose_msg.orientation.w = currentPose.Rot().W();
    pose_msg.orientation.x = currentPose.Rot().X();
    pose_msg.orientation.y = currentPose.Rot().Y();
    pose_msg.orientation.z = currentPose.Rot().Z();
    this->pose_pub_->publish(pose_msg);

    // Calculate time step
    std::chrono::duration<double> dtDuration = info.simTime - this->lastPoseUpdate;
    double dt = dtDuration.count();
    // RCLCPP_INFO(this->ros_node_->get_logger(), "passed time in seconds: %f", dt); // 0.004s
    
    this->lastPoseUpdate = info.simTime;

    // Ensure we have a reasonable dt to avoid division by zero or huge velocities
    if (dt < 1e-6)
        return;

    if (movement_enabled_) {
        // Get the target position and apply noise
        gz::math::Vector3d target_position = atk_target_pose_.Pos();
        gz::math::Vector3d noisy_position = ApplyGaussianNoise(target_position);
        
        // Move the current pose to the noisy target position
        currentPose.Pos() = noisy_position;
        *poseComp = gz::sim::components::Pose(currentPose);
        ecm.SetChanged(this->actorEntity_,
            gz::sim::components::Pose::typeId, 
            gz::sim::ComponentState::OneTimeChange);
        
        current_velocity_ = gz::math::Vector3d::Zero;
    }
    
    // Publish the current pose and velocity
    geometry_msgs::msg::TwistStamped pose_vel_msg;
    pose_vel_msg.header.stamp = this->ros_node_->now();
    pose_vel_msg.header.frame_id = "world";
    pose_vel_msg.twist.linear.x = current_velocity_.X();
    pose_vel_msg.twist.linear.y = current_velocity_.Y();
    pose_vel_msg.twist.linear.z = current_velocity_.Z();
    pose_vel_msg.twist.angular.x = currentPose.Pos().X();
    pose_vel_msg.twist.angular.y = currentPose.Pos().Y();
    pose_vel_msg.twist.angular.z = currentPose.Pos().Z();
    this->pose_vel_pub_->publish(pose_vel_msg);
}

GZ_ADD_PLUGIN(
    gazebo_sim_plugin::ActorPositionControllerPlugin,
    gz::sim::System,
    gazebo_sim_plugin::ActorPositionControllerPlugin::ISystemConfigure,
    gazebo_sim_plugin::ActorPositionControllerPlugin::ISystemPreUpdate
)