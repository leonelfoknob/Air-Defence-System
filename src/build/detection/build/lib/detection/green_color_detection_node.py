import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class GreenColorDetectionNode(Node):
    def __init__(self):
        super().__init__('green_color_detection_node')

        # Initialize publisher for both topics
        self.image_pub = self.create_publisher(Image, '/green_color_detect', 10)
        self.center_pub = self.create_publisher(Point, '/green_color_center', 10)

        # Initialize CvBridge to convert ROS image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the /image topic to get the video stream
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Define the center and limit for drawing reference lines
        self.limit = 50
        self.center_x = 0
        self.center_y = 0

    def adjust_hsv_ranges(self, frame):
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightness_factor = avg_brightness / 128  # Adjust brightness
        
        lower_green = np.array([40, int(40 * brightness_factor), int(40 * brightness_factor)])
        upper_green = np.array([80, 255, 255])
        
        return lower_green, upper_green

    def image_callback(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Get the frame width and height
        self.center_x = int(frame.shape[1] / 2)
        self.center_y = int(frame.shape[0] / 2)

        # Convert to HSV
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green, upper_green = self.adjust_hsv_ranges(frame)

        # Create a mask for green color
        Green_mask = cv2.inRange(HSV, lower_green, upper_green)
        Green_mask = cv2.erode(Green_mask, None, iterations=2)
        Green_mask = cv2.dilate(Green_mask, None, iterations=2)
        
        # Bitwise operation to get the green regions
        green_result = cv2.bitwise_and(frame, frame, mask=Green_mask)

        contours_green, _ = cv2.findContours(Green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours_green, _ = cv2.findContours(green_result , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_green_area = 0
        max_green_contour = None
        total_green_area = sum(cv2.contourArea(contour) for contour in contours_green)
        
        for contour in contours_green:
            area = cv2.contourArea(contour)
            
            if area > 500:
                if area > max_green_area:
                    max_green_area = area
                    max_green_contour = contour
        
        if max_green_contour is not None:
            x, y, w, h = cv2.boundingRect(max_green_contour)
            center_x_obj = x + w // 2
            center_y_obj = y + h // 2
            center = (x + w // 2, y + h // 2)

            # Draw bounding box and center on the frame
            cv2.circle(frame, (center_x_obj, center_y_obj), 5, (0, 255, 0), -1)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"({center_x_obj}, {center_y_obj})", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw reference lines
        cv2.line(frame, (self.center_x + self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x - self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y + self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y - self.limit), (0, 255, 0), 2)
        
        # Publish green color center
        if max_green_contour is not None:
            center_point = Point()
            center_point.x = float(center_x_obj)  # Ensure it's a float
            center_point.y = float(center_y_obj)  # Ensure it's a float
            self.center_pub.publish(center_point)
        
        # Publish the frame with detected green areas
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GreenColorDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()




'''import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class GreenColorDetectionNode(Node):
    def __init__(self):
        super().__init__('green_color_detection_node')

        # Initialize publisher for both topics
        self.image_pub = self.create_publisher(Image, '/green_color_detect', 10)
        self.center_pub = self.create_publisher(Point, '/green_color_center', 10)

        # OpenCV video capture
        self.cam = cv2.VideoCapture(0)
        
        # Get the frame width and height
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Center of the frame
        self.center_x = int(self.frame_width / 2)
        self.center_y = int(self.frame_height / 2)
        self.point = (self.center_x, self.center_y)
        
        self.limit = 50
        self.bridge = CvBridge()

        # Timer for capturing frames periodically
        self.timer = self.create_timer(0.1, self.timer_callback)

    def adjust_hsv_ranges(self, frame):
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightness_factor = avg_brightness / 128  # Adjust brightness
        
        lower_green = np.array([40, int(40 * brightness_factor), int(40 * brightness_factor)])
        upper_green = np.array([80, 255, 255])
        
        return lower_green, upper_green

    def timer_callback(self):
        ret, frame = self.cam.read()

        if not ret:
            self.get_logger().error('Failed to capture frame')
            return
        
        # Convert to HSV
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green, upper_green = self.adjust_hsv_ranges(frame)

        # Create a mask for green color
        Green_mask = cv2.inRange(HSV, lower_green, upper_green)
        Green_mask = cv2.erode(Green_mask, None, iterations=2)
        Green_mask = cv2.dilate(Green_mask, None, iterations=2)
        
        # Bitwise operation to get the green regions
        green_result = cv2.bitwise_and(frame, frame, mask=Green_mask)

        contours_green, _ = cv2.findContours(Green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_green_area = 0
        max_green_contour = None
        total_green_area = sum(cv2.contourArea(contour) for contour in contours_green)
        
        for contour in contours_green:
            area = cv2.contourArea(contour)
            
            if area > 500:
                if area > max_green_area:
                    max_green_area = area
                    max_green_contour = contour
        
        if max_green_contour is not None:
            x, y, w, h = cv2.boundingRect(max_green_contour)
            center_x_obj = x + w // 2
            center_y_obj = y + h // 2
            center = (x + w // 2, y + h // 2)

            # Draw bounding box and center on the frame
            cv2.circle(frame, (center_x_obj, center_y_obj), 5, (0, 255, 0), -1)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"({center_x_obj}, {center_y_obj})", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw reference lines
        cv2.line(frame, (self.center_x + self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x - self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y + self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y - self.limit), (0, 255, 0), 2)
        
        # Publish green color center
        if max_green_contour is not None:
            center_point = Point()
            center_point.x = float(center_x_obj)  # Ensure it's a float
            center_point.y = float(center_y_obj)  # Ensure it's a float
            self.center_pub.publish(center_point)
        
        # Publish the frame with detected green areas
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GreenColorDetectionNode()
    rclpy.spin(node)
    node.cam.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
'''