import subprocess
import time
import json


class TopicBridgeManager:
    """
    This class registers and stores all Gazebo topics, and uses `ros_gz_bridge`
    to create ROS2-Gazebo bridges for necessary topics during simulation.
    """

    def __init__(self):
        # Dictionary to store topics and their bridges
        self.gazebo_topics = {}
        self.ros2_topics = set()
        self.bridges = {}  # Stores subprocesses for active bridges
        self.register_gazebo_topics()
        self.register_ros2_topics()
        self.bridge_config = json.load(open("./utils/gazebo_ros_bridge_config.json"))

    def register_gazebo_topics(self):
        """
        Registers all available Gazebo topics by calling `gz topic -l`.
        """
        try:
            result = subprocess.run(
                ["gz", "topic", "-l"], capture_output=True, text=True, check=True
            )
            topic_list = result.stdout.splitlines()
            for topic in topic_list:
                self.gazebo_topics[topic] = None  # Initially no bridge for each topic
        except subprocess.CalledProcessError:
            print("Error: Unable to retrieve Gazebo topics.")

    def register_ros2_topics(self):
        """
        Registers all available ROS2 topics by calling `ros2 topic list`.
        """
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"], capture_output=True, text=True, check=True
            )
            topic_list = result.stdout.splitlines()
            self.ros2_topics = set(topic_list)
        except subprocess.CalledProcessError:
            print("Error: Unable to retrieve ROS2 topics.")

    def start_bridge(self, topic, ros2_type, gz_type=None):
        """
        Starts a ROS-Gazebo bridge for a specified topic with the given message types.

        Parameters:
        - topic (str): The Gazebo topic to bridge.
        - ros2_type (str): The ROS2 message type.
        - gz_type (str): The Gazebo message type.

        Returns:
        - bool: True if the bridge is started successfully, False otherwise.
        """

        if topic in self.gazebo_topics and self.gazebo_topics[topic] is None:
            if gz_type is None:
                gz_type = self.bridge_config.get("topic_message_types", {}).get(
                    ros2_type
                )

            command = [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                f"{topic}@{ros2_type}@{gz_type}",
            ]
            process = subprocess.Popen(command)
            self.bridges[topic] = process
            self.gazebo_topics[topic] = {
                "ros2_type": ros2_type,
                "gz_type": gz_type,
                "process": process,
            }
            self.register_ros2_topics()

            return True
        else:
            print(f"Bridge for topic '{topic}' already exists or topic not found.")
            return False

    def stop_bridge(self, topic):
        """
        Stops the bridge for a specified topic if it is active.

        Parameters:
        - topic (str): The topic for which to stop the bridge.

        Returns:
        - bool: True if the bridge is stopped successfully, False otherwise.
        """
        if topic in self.bridges:
            self.bridges[topic].terminate()
            self.bridges[topic].wait()
            del self.bridges[topic]
            self.gazebo_topics[topic] = None
            self.register_ros2_topics()

            print(f"Stopped bridge for topic '{topic}'.")
            return True
        else:
            print(f"No active bridge found for topic '{topic}'.")
            return False

    def list_active_bridges(self):
        """
        Lists all active bridges.

        Returns:
        - list: A list of topics that currently have active bridges.
        """
        return list(self.bridges.keys())

    def start_common_bridges(self):
        """
        Start common bridges for essential topics like camera and camera info.
        """
        # Example: Start bridges for camera and camera info if available
        if "/camera" in self.gazebo_topics and "/camera" not in self.ros2_topics:
            self.start_bridge("/camera", "sensor_msgs/msg/Image", "gz.msgs.Image")

        if "/camera_info" in self.gazebo_topics and "/camera_info" not in self.ros2_topics:
            self.start_bridge(
                "/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"
            )

        # if "/boxes_visible_2d_image" in self.gazebo_topics and "/boxes_visible_2d_image" not in self.ros2_topics:
        #     self.start_bridge(
        #         "/boxes_visible_2d_image", "sensor_msgs/msg/Image", "gz.msgs.Image"
        #     )
        # if "/boxes_full_2d_image" in self.gazebo_topics and "/boxes_full_2d_image" not in self.ros2_topics:
        #     self.start_bridge(
        #         "/boxes_full_2d_image", "sensor_msgs/msg/Image", "gz.msgs.Image"
        #     )
        # if "/boxes_3d_image" in self.gazebo_topics and "/boxes_3d_image" not in self.ros2_topics:
        #     self.start_bridge(
        #         "/boxes_3d_image", "sensor_msgs/msg/Image", "gz.msgs.Image"
        #     )

        if "/boxes_visible_2d" in self.gazebo_topics and "/boxes_visible_2d" not in self.ros2_topics:
            self.start_bridge(
                "/boxes_visible_2d",
                "vision_msgs/msg/Detection2DArray",
                "gz.msgs.AnnotatedAxisAligned2DBox_V",
            )
        # if "/boxes_full_2d" in self.gazebo_topics and "/boxes_full_2d" not in self.ros2_topics:
        #     self.start_bridge(
        #         "/boxes_full_2d",
        #         "vision_msgs/msg/Detection2DArray",
        #         "gz.msgs.AnnotatedAxisAligned2DBox_V",
        #     )
        if "/boxes_3d" in self.gazebo_topics and "/boxes_3d" not in self.ros2_topics:
            self.start_bridge(
                "/boxes_3d",
                "vision_msgs/msg/Detection3DArray",
                "gz.msgs.AnnotatedOriented3DBox_V",
            )

        # if "/depth_camera" in self.gazebo_topics and "/depth_camera" not in self.ros2_topics:
        #     self.start_bridge(
        #         "/depth_camera", "sensor_msgs/msg/Image", "gz.msgs.Image"
        #     )
            
        if "/world/default/pose/info" in self.gazebo_topics and "/world/default/pose/info" not in self.ros2_topics:
            self.start_bridge(
                "/world/default/pose/info",
                "geometry_msgs/msg/PoseArray",
                "gz.msgs.Pose_V"
            )
        
        # if "/world/default/dynamic_pose/info" in self.gazebo_topics:
        #     self.start_bridge(
        #         "/world/default/dynamic_pose/info",
        #         "geometry_msgs/msg/PoseArray",
        #         "gz.msgs.Pose_V"
        #     )
        
        if "/atk1_camera" in self.gazebo_topics and "/atk1_camera" not in self.ros2_topics:
            self.start_bridge("/atk1_camera", "sensor_msgs/msg/Image", "gz.msgs.Image")
        if "/atk1_camera_vis_2d" in self.gazebo_topics and "/atk1_camera_vis_2d" not in self.ros2_topics:
            self.start_bridge(
                "/atk1_camera_vis_2d",
                "vision_msgs/msg/Detection2DArray",
                "gz.msgs.AnnotatedAxisAligned2DBox_V",
            )
        if "/atk1_camera_3d" in self.gazebo_topics and "/atk1_camera_3d" not in self.ros2_topics:
            self.start_bridge(
                "/atk1_camera_3d",
                "vision_msgs/msg/Detection3DArray",
                "gz.msgs.AnnotatedOriented3DBox_V",
            )

        # wait for bridges to start
        time.sleep(3)

    def stop_all_bridges(self):
        """
        Stops all active bridges.
        """
        for topic in list(self.bridges.keys()):
            self.stop_bridge(topic)
        print("Stopped all active bridges.")

    def __str__(self) -> str:
        str_repr = "Active bridges:\n"
        for topic in self.bridges:
            str_repr += f"- {topic}\n"
        return str_repr


def main():
    # Example usage
    manager = TopicBridgeManager()
    manager.start_common_bridges()

    print()
    print(manager)

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping all bridges...")
    manager.stop_all_bridges()


if __name__ == "__main__":
    main()
