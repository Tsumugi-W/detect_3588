"""YOLOv5 ONNX Runtime inference adapter used by the ROS1 node.

The preprocessing and postprocessing here intentionally mirror the current
``detect_3588-main`` detector.  The small :class:`Detection` adapter is kept so
the ROS1 node and the visual-servo pipeline do not need to change.
"""
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Sequence[float]

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _letterbox(image, size, color=(114, 114, 114)):
    height, width = image.shape[:2]
    gain = min(float(size) / height, float(size) / width)
    resized = (int(round(width * gain)), int(round(height * gain)))
    pad_x = (size - resized[0]) / 2.0
    pad_y = (size - resized[1]) / 2.0
    if (width, height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, gain, pad_x, pad_y


def _nms(boxes, scores, threshold):
    """Return indices kept by the source detector's deterministic NMS."""

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = (
            np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        )
        union = areas[index] + areas[order[1:]] - intersection
        overlap = intersection / np.maximum(union, 1e-9)
        order = order[np.where(overlap <= float(threshold))[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def _postprocess(
    output,
    image_shape,
    input_size,
    confidence_threshold,
    iou_threshold,
):
    """Decode a decoded YOLOv5 output and apply class and cross-class NMS."""

    prediction = output[0] if output.ndim == 3 else output
    if prediction.size == 0:
        return []

    prediction = prediction[prediction[:, 4] > confidence_threshold]
    if len(prediction) == 0:
        return []

    class_scores = prediction[:, 5:] * prediction[:, 4:5]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]
    accepted = confidences > confidence_threshold
    prediction = prediction[accepted]
    class_ids = class_ids[accepted]
    confidences = confidences[accepted]
    if len(prediction) == 0:
        return []

    boxes = np.empty((len(prediction), 4), dtype=np.float32)
    boxes[:, 0] = prediction[:, 0] - prediction[:, 2] * 0.5
    boxes[:, 1] = prediction[:, 1] - prediction[:, 3] * 0.5
    boxes[:, 2] = prediction[:, 0] + prediction[:, 2] * 0.5
    boxes[:, 3] = prediction[:, 1] + prediction[:, 3] * 0.5

    image_height, image_width = image_shape
    gain = min(float(input_size) / image_height, float(input_size) / image_width)
    pad_x = (input_size - image_width * gain) / 2.0
    pad_y = (input_size - image_height * gain) / 2.0
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)

    kept_boxes = []
    kept_scores = []
    kept_ids = []
    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_confidences = confidences[class_mask]
        keep = _nms(class_boxes, class_confidences, iou_threshold)
        kept_boxes.append(class_boxes[keep])
        kept_scores.append(class_confidences[keep])
        kept_ids.append(np.full(len(keep), class_id, dtype=np.int32))

    boxes = np.concatenate(kept_boxes)
    confidences = np.concatenate(kept_scores)
    class_ids = np.concatenate(kept_ids)
    keep = _nms(boxes, confidences, iou_threshold)
    return [
        (boxes[index], float(confidences[index]), int(class_ids[index]))
        for index in keep
    ]


class YoloV5Onnx:
    def __init__(
        self,
        model_path,
        class_names,
        input_size=640,
        confidence_threshold=0.30,
        iou_threshold=0.01,
        threads=4,
    ):
        options = ort.SessionOptions()
        if int(threads) < 1:
            raise ValueError("ONNX threads must be at least 1")
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

        # Match the RK3588 detector: initialize graph optimizations and memory
        # before the first camera frame reaches the synchronized callback.
        dummy = np.zeros(
            (1, 3, self.input_size, self.input_size), dtype=np.float32
        )
        self.session.run(None, {self.input_name: dummy})

    def detect(self, image) -> List[Detection]:
        padded, _, _, _ = _letterbox(image, self.input_size)
        tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        tensor = tensor[np.newaxis, ...]
        output = self.session.run(None, {self.input_name: tensor})[0]
        detections = []
        decoded = _postprocess(
            output,
            image.shape[:2],
            self.input_size,
            self.confidence_threshold,
            self.iou_threshold,
        )
        for bbox, confidence, class_id in decoded:
            name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else str(class_id)
            )
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    confidence=confidence,
                    bbox=[float(value) for value in bbox],
                )
            )
        return detections
