'''
# version 1 ...

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class GreenColorFollower(Node):
    def __init__(self):
        super().__init__('green_color_follow')

        # Define the center and limit for drawing reference lines
        self.limit = 50
        self.center_x = 0
        self.center_y = 0

        self.color_center_x = 0
        self.color_center_y = 0

        self.command_str_x = ""
        self.command_str_y = ""
        
        # Subscribe to green color center
        self.center_sub = self.create_subscription(Point, '/green_color_center', self.center_callback, 10)
        
        # Subscribe to green color image
        self.image_sub = self.create_subscription(Image, '/green_color_detect', self.image_callback, 10)
        
        # OpenCV bridge
        self.bridge = CvBridge()

    def center_callback(self, msg):

        self.color_center_x = msg.x
        self.color_center_y = msg.y

        self.get_logger().info(f"Green color center at: x={msg.x}, y={msg.y}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Get the frame width and height
            self.center_x = int(cv_image.shape[1] / 2)
            self.center_y = int(cv_image.shape[0] / 2)

            self.up_limit = self.center_y - self.limit
            self.down_limit = self.center_y + self.limit
            self.left_limit =  self.center_x - self.limit
            self.right_limit = self.center_x + self.limit

            if self.color_center_x <= self.left_limit:
                self.command_str_x = f"turn right"

            elif self.color_center_x >= self.right_limit:
                self.command_str_x = f"turn left"
            
            else:
                self.command_str_x = f"center_x"
            
            if self.color_center_y >= self.down_limit:
                self.command_str_y = f"go up"
            
            elif self.color_center_y <= self.up_limit:
                self.command_str_y = f"go down"
            
            else:
                self.command_str_y = f"censter_y"

            self.get_logger().info(self.command_str_x)
            self.get_logger().info(self.command_str_y)
            
            # Display image
            cv2.imshow("Green Color Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = GreenColorFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
'''

'''
# version 2 .....
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time  # Import time module for timestamp handling

class GreenColorFollower(Node):
    def __init__(self):
        super().__init__('green_color_follow')

        # Define the center and limit for drawing reference lines
        self.limit = 50
        self.center_x = 0
        self.center_y = 0

        self.color_center_x = 0
        self.color_center_y = 0

        self.command_str_x = ""
        self.command_str_y = ""
        self.command_str_lazer = ""

        self.control_x =0
        self.control_y =0
        self.control_lazer = 0
        
        # Last update timestamp
        self.last_update_time = time.time()

        # Subscribe to green color center
        self.center_sub = self.create_subscription(Point, '/green_color_center', self.center_callback, 10)
        
        # Subscribe to green color image
        self.image_sub = self.create_subscription(Image, '/green_color_detect', self.image_callback, 10)
        
        # OpenCV bridge
        self.bridge = CvBridge()

    def center_callback(self, msg):
        # Update the center coordinates and timestamp
        self.color_center_x = msg.x
        self.color_center_y = msg.y
        self.last_update_time = time.time()  # Update last received time

        self.get_logger().info(f"Green color center at: x={msg.x}, y={msg.y}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Get the frame width and height
            self.center_x = int(cv_image.shape[1] / 2)
            self.center_y = int(cv_image.shape[0] / 2)

            self.up_limit = self.center_y - self.limit
            self.down_limit = self.center_y + self.limit
            self.left_limit =  self.center_x - self.limit
            self.right_limit = self.center_x + self.limit

            # Check if 5 seconds have passed since the last update
            if time.time() - self.last_update_time > 1:# 1 saniye sonra güncel veri yoksa hiç bir commut göndermiyor yada stop komut gönderiyor
                self.command_str_x = "NOT X : "
                self.command_str_y = "NOT Y : "
                self.command_str_lazer = "Lazer Down : "
                self.control_x =0
                self.control_y =0
                self.control_lazer = 0
            else:
                if self.color_center_x < self.right_limit and self.color_center_x > self.left_limit and self.color_center_y < self.down_limit and self.color_center_y > self.up_limit:
                    self.command_str_x = "center_x : "
                    self.command_str_y = "center_y : "
                    self.command_str_lazer = "Lazer up Fire !!! : "
                    self.control_x =0
                    self.control_y =0
                    self.control_lazer = 1
                else:


                    if self.color_center_x <= self.left_limit:
                        self.command_str_x = "turn right : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x =-1
                        self.control_lazer = 0
                    elif self.color_center_x >= self.right_limit:
                        self.command_str_x = "turn left : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x =1
                        self.control_lazer = 0
                    else:
                        self.command_str_x = "center_x : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x =0
                        self.control_lazer = 0

                    if self.color_center_y >= self.down_limit:
                        self.command_str_y = "go up : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y =-1
                        self.control_lazer = 0
                    elif self.color_center_y <= self.up_limit:
                        self.command_str_y = "go down : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y =1
                        self.control_lazer = 0
                    else:
                        self.command_str_y = "center_y : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y =0
                        self.control_lazer = 0

            self.get_logger().info(self.command_str_x+f"{self.control_x}")
            self.get_logger().info(self.command_str_y+f"{self.control_y}")
            self.get_logger().info(self.command_str_lazer+f"{self.control_lazer}")
            
            # Display image
            cv2.imshow("Green Color Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = GreenColorFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()'''

#version 3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Int32  # Import Int32 message type for publishing commands
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time  # Import time module for timestamp handling

class GreenColorFollower(Node):
    def __init__(self):
        super().__init__('green_color_follow')

        # Define the center and limit for drawing reference lines
        self.limit = 50
        self.center_x = 0
        self.center_y = 0

        self.color_center_x = 0
        self.color_center_y = 0

        self.command_str_x = ""
        self.command_str_y = ""
        self.command_str_lazer = ""

        self.control_x = 0
        self.control_y = 0
        self.control_lazer = 0
        
        # Last update timestamp
        self.last_update_time = time.time()

        # Subscribe to green color center
        self.center_sub = self.create_subscription(Point, '/green_color_center', self.center_callback, 10)
        
        # Subscribe to green color image
        self.image_sub = self.create_subscription(Image, '/green_color_detect', self.image_callback, 10)
        
        # OpenCV bridge
        self.bridge = CvBridge()

        # *** Add Publishers ***
        self.x_pub = self.create_publisher(Int32, '/x_axis_command', 10)
        self.y_pub = self.create_publisher(Int32, '/y_axis_command', 10)
        self.lazer_pub = self.create_publisher(Int32, '/lazer_command', 10)

    def center_callback(self, msg):
        # Update the center coordinates and timestamp
        self.color_center_x = msg.x
        self.color_center_y = msg.y
        self.last_update_time = time.time()  # Update last received time

        self.get_logger().info(f"Green color center at: x={msg.x}, y={msg.y}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Get the frame width and height
            self.center_x = int(cv_image.shape[1] / 2)
            self.center_y = int(cv_image.shape[0] / 2)

            self.up_limit = self.center_y - self.limit
            self.down_limit = self.center_y + self.limit
            self.left_limit =  self.center_x - self.limit
            self.right_limit = self.center_x + self.limit

            # Check if 5 seconds have passed since the last update
            if time.time() - self.last_update_time > 1:
                self.command_str_x = "NOT X : "
                self.command_str_y = "NOT Y : "
                self.command_str_lazer = "Lazer Down : "
                self.control_x = -10
                self.control_y = -10
                self.control_lazer = 0
            else:
                if self.color_center_x < self.right_limit and self.color_center_x > self.left_limit and self.color_center_y < self.down_limit and self.color_center_y > self.up_limit:
                    self.command_str_x = "center_x : "
                    self.command_str_y = "center_y : "
                    self.command_str_lazer = "Lazer up Fire !!! : "
                    self.control_x = 0
                    self.control_y = 0
                    self.control_lazer = 1
                else:
                    if self.color_center_x <= self.left_limit:
                        self.command_str_x = "turn right : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x = -1
                        self.control_lazer = 0
                    elif self.color_center_x >= self.right_limit:
                        self.command_str_x = "turn left : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x = 1
                        self.control_lazer = 0
                    else:
                        self.command_str_x = "center_x : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_x = 0
                        self.control_lazer = 0

                    if self.color_center_y >= self.down_limit:
                        self.command_str_y = "go up : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y = -1
                        self.control_lazer = 0
                    elif self.color_center_y <= self.up_limit:
                        self.command_str_y = "go down : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y = 1
                        self.control_lazer = 0
                    else:
                        self.command_str_y = "center_y : "
                        self.command_str_lazer = "Lazer Down : "
                        self.control_y = 0
                        self.control_lazer = 0

            self.get_logger().info(self.command_str_x + f"{self.control_x}")
            self.get_logger().info(self.command_str_y + f"{self.control_y}")
            self.get_logger().info(self.command_str_lazer + f"{self.control_lazer}")

            # *** Publish control values ***
            self.x_pub.publish(Int32(data=self.control_x))
            self.y_pub.publish(Int32(data=self.control_y))
            self.lazer_pub.publish(Int32(data=self.control_lazer))
            
            # Display image
            cv2.imshow("Green Color Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = GreenColorFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
