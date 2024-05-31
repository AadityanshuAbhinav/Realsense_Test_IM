#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np

class IntelPublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            self.profile = self.pipeline.start(self.cfg)
        except Exception as e:
            self.get_logger().error(f"Failed to start the pipeline: {e}")
            return
        
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.timer = self.create_timer(0.1, self.capture_and_publish)  # 10 Hz

    def capture_and_publish(self):
        while rclpy.ok():
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame or not depth_frame:
                self.get_logger().warning("No frames received")
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            # print(depth_image)
            color_image = np.asanyarray(color_frame.get_data())
            
            ros_rgb_image = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")
            # print(ros_rgb_image)
            
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = self.get_clock().now().to_msg()
            camera_info_msg.header.frame_id = "camera_color_optical_frame"
            camera_info_msg.width = self.color_intrinsics.width
            camera_info_msg.height = self.color_intrinsics.height
            camera_info_msg.k = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0, 0.0, 1.0]
            camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
            camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            camera_info_msg.p = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0, 0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            self.rgb_pub.publish(ros_rgb_image)
            self.depth_pub.publish(ros_depth_image)
            self.camera_info_pub.publish(camera_info_msg)
            # print(camera_info_msg.k)
            self.get_logger().info("Published images and camera info ")

        # try:
        #     frames = self.pipeline.wait_for_frames()
        #     depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()

        #     if not color_frame or not depth_frame:
        #         return

        #     depth_image = np.asanyarray(depth_frame.get_data())
        #     color_image = np.asanyarray(color_frame.get_data())

        #     ros_rgb_image = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        #     ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")

        #     camera_info_msg = CameraInfo()
        #     camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        #     camera_info_msg.header.frame_id = "camera_color_optical_frame"
        #     camera_info_msg.width = self.color_intrinsics.width
        #     camera_info_msg.height = self.color_intrinsics.height
        #     camera_info_msg.k = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0, 0.0, 1.0]
        #     camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        #     camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        #     camera_info_msg.p = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0, 0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]

        #     self.rgb_pub.publish(ros_rgb_image)
        #     self.depth_pub.publish(ros_depth_image)
        #     self.camera_info_pub.publish(camera_info_msg)
        #     self.get_logger().info("Published images and camera info")

        # except Exception as e:
        #     self.get_logger().error(f"Error capturing and publishing frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    rs_publisher = IntelPublisher()
    try:
        rs_publisher.capture_and_publish()
    except KeyboardInterrupt:
        pass
    finally:
        rs_publisher.pipeline.stop()
        rclpy.shutdown()
    # rclpy.spin(rs_publisher)
    # rs_publisher.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
