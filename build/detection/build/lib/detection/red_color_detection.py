import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedColorDetectionNode(Node):
    def __init__(self):
        super().__init__('red_color_detection_node')

        # Initialize publisher for both topics
        self.image_pub = self.create_publisher(Image, '/red_color_detect', 10)
        self.center_pub = self.create_publisher(Point, '/red_color_center', 10)

        # Initialize CvBridge to convert ROS image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the /image topic to get the video stream
        self.create_subscription(Image, '/image', self.image_callback, 10)

        # Define the center and limit for drawing reference lines
        self.limit = 50
        self.center_x = 0
        self.center_y = 0

    def adjust_hsv_ranges(self, frame):
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightness_factor = avg_brightness / 128  # Adjust brightness
        
        # Red color ranges in HSV space
        lower_red1 = np.array([0, 150, 100])  
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 150, 100])  
        upper_red2 = np.array([180, 255, 255])

        return lower_red1, lower_red2, upper_red1, upper_red2

    def image_callback(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Get the frame width and height
        self.center_x = int(frame.shape[1] / 2)
        self.center_y = int(frame.shape[0] / 2)

        # Convert to HSV
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1, lower_red2, upper_red1, upper_red2 = self.adjust_hsv_ranges(frame)

        # Create a mask for red color
        Red_mask1 = cv2.inRange(HSV, lower_red1, upper_red1)
        Red_mask2 = cv2.inRange(HSV, lower_red2, upper_red2)
        Red_mask = Red_mask1 | Red_mask2  # Combine the two red masks
        Red_mask = cv2.erode(Red_mask, None, iterations=2)
        Red_mask = cv2.dilate(Red_mask, None, iterations=2)
        
        # Bitwise operation to get the red regions
        red_result = cv2.bitwise_and(frame, frame, mask=Red_mask)

        contours_red, _ = cv2.findContours(Red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_red_area = 0
        max_red_contour = None
        total_red_area = sum(cv2.contourArea(contour) for contour in contours_red)
        
        for contour in contours_red:
            area = cv2.contourArea(contour)
            
            if area > 500:
                if area > max_red_area:
                    max_red_area = area
                    max_red_contour = contour
        
        if max_red_contour is not None:
            x, y, w, h = cv2.boundingRect(max_red_contour)
            center_x_obj = x + w // 2
            center_y_obj = y + h // 2
            center = (x + w // 2, y + h // 2)

            # Draw bounding box and center on the frame
            cv2.circle(frame, (center_x_obj, center_y_obj), 5, (0, 0, 255), -1)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"({center_x_obj}, {center_y_obj})", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw reference lines
        cv2.line(frame, (self.center_x + self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x - self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y + self.limit), (self.center_x + self.limit, self.center_y + self.limit), (0, 255, 0), 2)
        cv2.line(frame, (self.center_x - self.limit, self.center_y - self.limit), (self.center_x + self.limit, self.center_y - self.limit), (0, 255, 0), 2)
        
        # Publish red color center
        if max_red_contour is not None:
            center_point = Point()
            center_point.x = float(center_x_obj)  # Ensure it's a float
            center_point.y = float(center_y_obj)  # Ensure it's a float
            self.center_pub.publish(center_point)
        
        # Publish the frame with detected red areas
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RedColorDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
