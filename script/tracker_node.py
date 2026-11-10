#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv_bridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

if not hasattr(np, "bool"):
    np.bool = bool

np.bool = bool
np.object = object


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        yolo_model = self.param("yolo_model", "yolov8n.pt")
        self.input_topic = self.param("input_topic", "image_raw")
        self.result_topic = self.param("result_topic", "yolo_result")
        self.result_image_topic = self.param("result_image_topic", "yolo_image")
        self.conf_thres = self.param("conf_thres", 0.25)
        self.iou_thres = self.param("iou_thres", 0.45)
        self.max_det = self.param("max_det", 300)
        self.classes = self.param("classes", None)
        self.device = self.param("device", None)
        self.result_conf = self.param("result_conf", True)
        self.result_line_width = self.param("result_line_width", None)
        self.result_font_size = self.param("result_font_size", None)
        self.result_font = self.param("result_font", "Arial.ttf")
        self.result_labels = self.param("result_labels", True)
        self.result_boxes = self.param("result_boxes", True)
        self.enabled = self.param("enabled", True)
        path = get_package_share_directory("ultralytics_ros")
        is_engine = yolo_model.lower().endswith(".engine")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        if not is_engine:
            self.model.fuse()
        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            1,
        )
        self.results_pub = self.create_publisher(YoloResult, self.result_topic, 1)
        self.result_image_pub = self.create_publisher(
            CompressedImage, self.result_image_topic, 1
        )
        self.enabled_pub = self.create_publisher(
            Bool,
            "~/enabled",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.lower().endswith(("-seg.pt", "-seg.engine"))
        self.enable_srv = self.create_service(SetBool, "~/enable", self.enable_callback)
        self._publish_enabled_status()

    def param(self, name, default):
        """A parameter of any type; an empty value counts as unset."""
        value = self.declare_parameter(
            name, default, ParameterDescriptor(dynamic_typing=True)
        ).value
        return default if value is None or value == "" else value

    def enable_callback(self, req, response):
        self.enabled = req.data
        self._publish_enabled_status()
        message = f"YOLO inference {self.result_topic} " + (
            "enabled" if self.enabled else "disabled"
        )
        self.get_logger().info(message)
        response.success = True
        response.message = message
        return response

    def _publish_enabled_status(self):
        self.enabled_pub.publish(Bool(data=bool(self.enabled)))

    def image_callback(self, msg):
        if not self.enabled:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results, msg.header)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.position.x = float(bbox[0])
            detection.bbox.center.position.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            # ROS 2 class_id is a string; consumers expect the integer id in it.
            hypothesis.hypothesis.class_id = str(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results, header):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_compressed_imgmsg(
            plotted_image, dst_format="jpg"
        )
        result_image_msg.header = header
        return result_image_msg

    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
