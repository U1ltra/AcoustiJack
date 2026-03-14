import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Pose, Point, TwistStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from vision_msgs.msg import Detection2DArray, Detection3DArray
from image_geometry import PinholeCameraModel
from gz_helper import GzClockSubscriber
import numpy as np
from transforms3d.euler import quat2euler

class RosReader(Node):
    def __init__(self):
        super().__init__("ros_reader")

        self.camera_pose_sub = self.create_subscription(
            Pose, "/camera/pose", self.camera_pose_callback, 10
        )
        self.camera_pose = []
        self.gz_clock = GzClockSubscriber()

    def camera_pose_callback(self, msg):
        pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        # Calculate Euler angles
        roll, pitch, yaw = quat2euler(
            [pose[6], pose[3], pose[4], pose[5]],  # wxyz
            axes="sxyz",
        )

        self.camera_pose.append([roll, pitch, yaw, self.gz_clock.get_sim_time()])
        self.get_logger().info(f"Camera Pose: {self.camera_pose[-1]}")
        
    def save_camera_pose(self, filename):
        if self.camera_pose:
            np.save(filename, np.array(self.camera_pose))
            self.get_logger().info(f"Camera pose saved to {filename}")
        else:
            self.get_logger().warning("No camera pose data to save.")


if __name__ == "__main__":
    rclpy.init()
    node = RosReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_camera_pose("results/res7_usenix/camera_pose.npy")
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

