import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ShapeDetection(Node):
    def __init__(self):
        super().__init__('shape_detection')
        self.bridge = CvBridge()
        
        # Subscribe to the '/red_mask' topic to get the binary mask from red color detection
        self.image_sub = self.create_subscription(Image, '/red_mask', self.image_callback, 10)
        
        # Publisher for the output image with shapes detected
        self.image_pub = self.create_publisher(Image, 'shapes/image', 10)
        
        # Adjustable perimeter threshold to filter out small objects
        self.min_perimeter = 20  
        
        self.get_logger().info("Shape detection node started")

    def image_callback(self, msg):
        # Convert the ROS image message to a NumPy array
        sz = (msg.height, msg.width)
        
        if msg.encoding == 'mono8':  # Mask is typically in mono8 encoding
            image = np.array(msg.data).reshape(sz)
        else:
            self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
            return
        
        output_image = self.detect_shapes(image)
        
        # Publish the result
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "bgr8"))

    def detect_shapes(self, mask):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a color image to draw the contours on
        output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < self.min_perimeter:
                continue  # Skip small objects

            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            shape = "Unknown"

            # Classify shapes based on the number of corners
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Square"
            else:
                shape = "Circle"
                # Detect circles using Hough Transform
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius < 10:  # Ignore very small circles
                    continue
            
            # Draw the detected contour and shape name
            cv2.drawContours(output_image, [approx], 0, (0, 255, 0), 2)
            cv2.putText(output_image, shape, tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return output_image

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
