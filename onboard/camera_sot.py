from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from geometry_msgs.msg import Point # Best message type for publishing object x, y location and height
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

from utils.topic_bridge_manager import TopicBridgeManager
from onboard.tracker import Tracker

import numpy as np


class Camera:
    def __init__(self, topic_manager: TopicBridgeManager):
        """
        Initialize the Camera class with a reference to the TopicBridgeManager.

        Args:
            topic_manager (TopicBridgeManager): An instance of TopicBridgeManager to retrieve topics.
        """
        self.camera_types = ["raw", "visible_2d", "full_2d", "3d"]
        self.topic_manager = topic_manager
        self.current_camera_topic = None
        self.subscribers = {}
        self.object_publisher = ObjectPublisher()

        self.blur_kernel = None  # TODO

        # Bounds drawing
        self.current_frame = None
        self.drawing = False
        self.ix = None
        self.iy = None
        self.x = None
        self.y = None
        self.bbox = None
        self.tracker = Tracker()
        self.target_height = None

    @property
    def topic(self):
        return self.current_camera_topic

    @property
    def img_feed(self):
        if self.current_camera_topic not in self.subscribers:
            self.subscribers[self.current_camera_topic] = ImageSubscriber(self)
        return self.subscribers[self.current_camera_topic]

    def find_topic_by_camera_type(self, camera_type: str):
        """
        Search for a topic containing the specified camera type.

        Args:
            camera_type (str): The type of camera ("visible_2d", "full_2d", or "3d").

        Returns:
            str: The matched topic name or None if no matching topic is found.
        """
        if camera_type not in self.camera_types:
            print(f"Error: Camera type '{camera_type}' is not recognized.")
            return None

        if camera_type == "raw":
            camera_type = "camera"

        # Find the first topic containing the camera type keyword
        for topic in self.topic_manager.ros2_topics:
            topic = topic.split("/")[-1]
            if camera_type in topic:
                self.current_camera_topic = topic

                print(f"Found topic for camera type '{camera_type}': {topic}")
                return topic

        print(f"No topic found for camera type '{camera_type}'.")
        return None

    def set_blur_kernel(self, kernel_size):
        return  # TODO
        """
        Set the blur kernel size.
        
        Args:
            kernel_size (tuple): A tuple (x, y) specifying the kernel size for blurring.
        """
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.blur_kernel = kernel_size
            print(f"Blur kernel set to: {self.blur_kernel}")
        else:
            print("Error: Invalid kernel size format. Use a tuple like (x, y).")

    def apply_blur(self, image):
        """
        Apply blur to an image if a blur kernel is set.

        Args:
            image (ndarray): The image to apply blur on.

        Returns:
            ndarray: The blurred image.
        """
        if self.blur_kernel:
            return cv2.blur(image, self.blur_kernel)
        return image

    def draw_rectangle(self, event, x, y, flags, param):
        self.x, self.y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox = (self.ix, self.iy, x, y)
            self.state = -1

    def calc_pwise_dist(self, coords):
        distances = []

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)

        return np.sort(np.unique(np.array(distances)))
    
    def process_image(self, image):
        """
        Process an image by applying blur and object tracking.
        Args:
            image (ndarray): The image to process.

        Returns:
            ndarray: The processed image.
        """

        image_h, image_w = image.shape[:2]
        image = self.apply_blur(image)
        
        if self.bbox:
            self.state += 1

            if self.state == 0:
                x1, y1, x2, y2 = self.bbox
                if (x1 == x2 or y1 == y2):
                    self.state = 0
                else:
                    print('Tracker initialized!')

                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    x2 = max(x1, x2)
                    y2 = max(y1, y2)
                    
                    cv2.imshow('Template', image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)])

                    x_c = (x1 + x2) // 2
                    y_c = (y1 + y2) // 2
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)

                    self.tracker.set_template(image, np.array([x_c, y_c]), np.array([w, h]))
                    coords = np.array(self.tracker.track(image)).reshape(-1, 2)
                    self.target_height = self.calc_pwise_dist(coords)[0]
            else:
                coords = np.array(self.tracker.track(image)).reshape(-1, 2)
                cv2.polylines(image, [coords], isClosed=True, color=(0, 255, 0), thickness=2)
                
                centroid = coords.mean(axis=0)
                cv2.circle(image, centroid.astype(int), radius=3, color=(0,255,0), thickness=-1)

                data = Point()
                data.x = centroid[0] - (image_w // 2)
                data.y = centroid[1] - (image_h // 2)
                data.z = self.target_height - self.calc_pwise_dist(coords)[0]

                self.object_publisher.publish_coords(data)

        if self.drawing:
            cv2.rectangle(image, (self.ix, self.iy), (self.x, self.y), color=(255, 0, 0), thickness=2)

        self.current_frame = image

    def __str__(self) -> str:
        return f"Camera: {self.current_camera_topic}\nBlur Kernel: {self.blur_kernel}"

class ObjectPublisher(Node):
    """
    Create an ObjectPublisher class, which is a subclass of the Node class.
    """

    def __init__(self):
        super().__init__("object_publisher")
        self.publisher = self.create_publisher(Point, 'object', 10)

    def publish_coords(self, data):
        self.publisher.publish(data)



class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """

    def __init__(self, camera):
        super().__init__("image_subscriber")

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image, camera.topic, self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        self.camera = camera

    def listener_callback(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info("Receiving video frame")

        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        # Object Tracking
        self.camera.process_image(frame)

        cv2.namedWindow('Camera')
        cv2.setMouseCallback('Camera', self.camera.draw_rectangle)

        # Show Results
        cv2.imshow('Camera', self.camera.current_frame)
        cv2.waitKey(1)

def main():
    topic_manager = TopicBridgeManager()
    camera = Camera(topic_manager)
    camera.find_topic_by_camera_type("visible_2d")
    camera.set_blur_kernel((5, 5))
    print(camera)