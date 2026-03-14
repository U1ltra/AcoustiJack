from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from geometry_msgs.msg import Point # Best message type for publishing object x, y location and height
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

from utils.topic_bridge_manager import TopicBridgeManager
from onboard.model import Model


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
        self.model = None

        # Tracking module
        self.current_frame = None
        self.detected = None
        self.tracking = None
        self.state = -1
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

    def set_model(self, model: Model):
        """
        Set the object detection model for the camera.

        Args:
            model (Model): The object detection model to use.
        """
        self.model = model

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
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.detected is not None:
            for obj in self.detected:
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.tracking = obj.id
                    print("Selected object with ID ", obj.id)
                    self.state = -1
                    break
            print("No object at position ", x, ", ", y)

    def process_image(self, image):
        """
        Process an image by applying blur and object detection.

        Args:
            image (ndarray): The image to process.

        Returns:
            ndarray: The processed image.
        """
        image_h, image_w = image.shape[:2]

        image = self.apply_blur(image)

        if self.model:
            results = self.model.track(image, persist=True)
            self.detected = results[0].boxes
            
            image = results[0].plot()

            if self.tracking is not None:
                for obj in self.detected:
                    if obj.id == self.tracking:
                        self.state += 1

                        if self.state == 0:
                            self.target_height = obj.xywh[0][3]

                        x_err = int((obj.xyxy[0][0] + obj.xyxy[0][2]) / 2) - (image_w // 2)
                        y_err = int((obj.xyxy[0][1] + obj.xyxy[0][3]) / 2) - (image_h // 2)
                        z_err = int(obj.xywh[0][3] - self.target_height)

                        print(x_err, ", ", y_err, ", ", z_err)

                        data = Point()
                        data.x = float(x_err)
                        data.y = float(y_err)
                        data.z = float(z_err)

                        self.object_publisher.publish_coords(data)

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
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
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
        cv2.setMouseCallback('Camera', self.camera.mouse_callback)

        # Show Results
        cv2.imshow('Camera', self.camera.current_frame)
        cv2.waitKey(1)

def main():
    topic_manager = TopicBridgeManager()
    camera = Camera(topic_manager)
    camera.find_topic_by_camera_type("visible_2d")
    camera.set_blur_kernel((5, 5))
    print(camera)
