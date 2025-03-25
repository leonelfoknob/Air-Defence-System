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
        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, 'shapes/image', 10)
        self.min_perimeter = 500  # Adjustable perimeter threshold to filter out small objects
        self.get_logger().info("Shape detection node started")

    def image_callback(self, msg):
        sz = (msg.height, msg.width)
        if msg.encoding == 'rgb8':
            image = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            image[:, :, 2] = np.array(msg.data[0::3]).reshape(sz)
            image[:, :, 1] = np.array(msg.data[1::3]).reshape(sz)
            image[:, :, 0] = np.array(msg.data[2::3]).reshape(sz)
        elif msg.encoding == 'mono8':
            image = np.array(msg.data).reshape(sz)
        else:
            self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
            return

        output_image = self.detect_shapes(image)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "bgr8"))

    def detect_shapes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < self.min_perimeter:
                continue  # Skip small objects

            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            shape = "Unknown"

            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Square"
            else:
                shape = "Circle"
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                           param1=50, param2=30, minRadius=5, maxRadius=100)
                if circles is None:
                    continue  # Skip if not confirmed

            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
            cv2.putText(image, shape, tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
