"""ROS image message packing that does not depend on cv_bridge type tables."""

import numpy as np


def bgr8_image_message(image, message_type, header):
    """Pack a uint8 BGR array into a sensor_msgs/Image-compatible message."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("bgr8 image must use uint8 pixels")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("bgr8 image must have shape (height, width, 3)")

    array = np.ascontiguousarray(array)
    message = message_type()
    message.header = header
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = array.tobytes()
    return message
