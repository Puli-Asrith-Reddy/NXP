# Copyright 2025 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
# (License headers remain the same)

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import WarehouseShelf

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import pkg_resources
import torch
import torchvision
import time
import yaml
import tflite_runtime.interpreter as tflite

QOS_PROFILE_DEFAULT = 5
PACKAGE_NAME = 'b3rb_ros_aim_india'
RED_COLOR = (0, 0, 255); BLUE_COLOR = (255, 0, 0); GREEN_COLOR = (0, 255, 0)

# --- Helper Functions (xywh2xyxy, non_max_suppression remain the same, including fixes) ---
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2; y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2; y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300, nm=0):
    assert 0 <= conf_thres <= 1 and 0 <= iou_thres <= 1
    if isinstance(prediction,(list,tuple)): prediction=prediction[0]
    device=prediction.device; mps="mps" in device.type
    if mps: prediction=prediction.cpu()
    bs=prediction.shape[0]; nc=prediction.shape[2]-nm-5; xc=prediction[..., 4]>conf_thres
    max_wh=7680; max_nms=30000; time_limit=0.5+0.05*bs; redundant=True; multi_label=nc>1; merge=False
    t=time.time(); mi=5+nc; output=[torch.zeros((0,6+nm), device=prediction.device)]*bs
    for xi, x in enumerate(prediction):
        x=x[xc[xi]];
        if not x.shape[0]: continue
        x[:, 5:]*=x[:, 4:5]; box=xywh2xyxy(x[:, :4]); mask=x[:, mi:]
        conf, j=x[:, 5:mi].max(1, keepdim=True); x=torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1)>conf_thres]
        if classes is not None: x=x[(x[:, 5:6]==torch.tensor(classes, device=x.device)).any(1)]
        n=x.shape[0];
        if not n: continue
        x=x[x[:, 4].argsort(descending=True)[:max_nms]]
        c=x[:, 5:6]*(0 if agnostic else max_wh); boxes, scores=x[:, :4]+c, x[:, 4]
        i=torchvision.ops.nms(boxes, scores, iou_thres); i=i[:max_det]
        output[xi]=x[i]
        if mps: output[xi]=output[xi].to(device)
        if (time.time()-t)>time_limit: break
    return output

class ObjectRecognizer(Node):
    """ Recognizes objects and publishes individual detections with probabilities for each frame."""
    def __init__(self):
        super().__init__('object_recognizer')

        self.allowed_objects = ['banana', 'car', 'clock', 'cup', 'horse', 'zebra','teddy bear','potted plant'] # Corrected list
        # No QR detector needed here

        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, QOS_PROFILE_DEFAULT)
        self.publisher_shelf_objects = self.create_publisher( # Publishes every valid frame
            WarehouseShelf, '/shelf_objects', QOS_PROFILE_DEFAULT)
        self.publisher_object_recog = self.create_publisher( # Keep debug publisher
            CompressedImage, "/debug_images/object_recog", QOS_PROFILE_DEFAULT)
        # No service needed

        # --- Load Model and Labels ---
        try:
            # (Load model logic remains the same)
            resource_name_coco = "../../../../share/ament_index/resource_index/coco.yaml"
            resource_path_coco = pkg_resources.resource_filename(PACKAGE_NAME, resource_name_coco)
            resource_name_yolo = "../../../../share/ament_index/resource_index/yolov5n-int8.tflite"
            resource_path_yolo = pkg_resources.resource_filename(PACKAGE_NAME, resource_name_yolo)
            with open(resource_path_coco) as f: self.label_names = yaml.load(f, Loader=yaml.FullLoader)['names']
            self.interpreter = tflite.Interpreter(model_path=resource_path_yolo)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info("YOLOv5 TFLite model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model or labels: {e}")
            self.label_names = []; self.interpreter = None

        # --- No internal memory needed for best frame ---

    # No reset_memory or service handler needed

    def publish_debug_image(self, publisher, image):
        """Publishes debug images."""
        if image is not None and image.size > 0:
            try: msg = CompressedImage(format="jpeg"); _, buf = cv2.imencode('.jpg', image); msg.data = buf.tobytes(); publisher.publish(msg)
            except Exception as e: self.get_logger().error(f"Failed to publish debug image: {e}")

    def camera_image_callback(self, message):
        """Processes images, filters, and publishes individual detections + probabilities."""
        
        if self.interpreter is None: return

        # --- Decode Image ---
        try:
            np_arr = np.frombuffer(message.data, np.uint8); image_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image_color is None: raise ValueError("Decoded image is None")
            height, width, _ = image_color.shape
        except Exception as e: self.get_logger().error(f"Failed to decode image: {e}"); return

        # --- Pre-process for YOLO ---
        try:
            input_size = self.input_details[0]['shape'][1]; image_resized = cv2.resize(image_color, (input_size, input_size))
            image_float = image_resized.astype(np.float32); image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
            image_normalized = image_rgb / 255.0; img_batch = np.expand_dims(image_normalized, axis=0)
        except Exception as e: self.get_logger().error(f"Failed during preprocessing: {e}"); return

        # --- Store current frame's results temporarily ---
        current_object_names = []    # To store ['car', 'car', 'clock']
        current_object_probs = []    # To store [95, 88, 75]

        # --- Invoke YOLO Inference ---
        try:
            # (Inference logic remains the same)
            input_details = self.input_details[0]; int8 = input_details["dtype"] == np.uint8
            if int8: scale, zp = input_details["quantization"]; img_quant=(img_batch/scale+zp).astype(np.uint8); self.interpreter.set_tensor(input_details["index"], img_quant)
            else: self.interpreter.set_tensor(input_details["index"], img_batch)
            self.interpreter.invoke()
            yolo_outputs = []
            for output_details in self.output_details:
                out_tensor = self.interpreter.get_tensor(output_details["index"])
                if int8: scale, zp = output_details["quantization"]; out_tensor=(out_tensor.astype(np.float32)-zp)*scale
                yolo_outputs.append(out_tensor)
        except Exception as e: self.get_logger().error(f"Failed during inference: {e}"); return

        # --- Process YOLO Output ---
        debug_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        try:
            for pred in yolo_outputs:
                w, h = self.input_details[0]["shape"][1:3]
                pred[0][..., :4] *= [w, h, w, h]; pred_tensor = torch.tensor(pred)
                nms_results = non_max_suppression(pred_tensor, conf_thres=0.25, iou_thres=0.45)
                for i, det in enumerate(nms_results):
                    if len(det):
                        for *xyxy, conf, cls in reversed(det):
                            object_name = self.label_names[int(cls)]; score_int = int(float(conf) * 100) # Prob as int
                            
                            # Filter 2: Allowed Object
                            if object_name not in self.allowed_objects: continue
                            
                            # Add individual detection to lists
                            current_object_names.append(object_name)
                            current_object_probs.append(score_int)
                            
                            # Draw on debug image
                            start_pt=(int(xyxy[0]),int(xyxy[1])); end_pt=(int(xyxy[2]),int(xyxy[3]))
                            cv2.rectangle(debug_image, start_pt, end_pt, GREEN_COLOR, 2)
                            cv2.putText(debug_image, f"{object_name} {score_int}%", start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN_COLOR, 2, cv2.LINE_AA)
        except Exception as e: self.get_logger().error(f"Failed processing YOLO output: {e}")

        # --- Apply Filter 1 ---
        total_objects = len(current_object_names) # Count individuals
        if total_objects >=7:
            # self.get_logger().debug(f"Frame skipped: Too many objects ({total_objects})")
            return # Don't publish if too many detections

        # --- Publish Debug Image ---
        debug_image_resized = cv2.resize(debug_image, (width, height))
        self.publish_debug_image(self.publisher_object_recog, debug_image_resized)

        # --- Create and Publish Message ---
        # No internal memory or comparison needed here. Publish every valid frame.
        shelf_objects_message = WarehouseShelf()
        shelf_objects_message.object_name = current_object_names   # e.g., ['car', 'car', 'clock']
        shelf_objects_message.object_count = current_object_probs  # e.g., [95, 88, 75]
        #self.get_logger().info(f"probabilities : {current_object_probs}")
        # qr_decoded is not set here
        
        self.publisher_shelf_objects.publish(shelf_objects_message)


def main(args=None):
    rclpy.init(args=args)
    object_recognizer = ObjectRecognizer()
    try: rclpy.spin(object_recognizer)
    except KeyboardInterrupt: pass
    finally: object_recognizer.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()