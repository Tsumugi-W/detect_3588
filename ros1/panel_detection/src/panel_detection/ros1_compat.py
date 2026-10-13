"""Small rclpy-like facade used by the ROS1 Noetic detector port."""

import rospy
import threading


class _ParameterValue:
    def __init__(self, value):
        self.string_value = str(value) if value is not None else ""
        self.bool_value = bool(value)
        try:
            self.integer_value = int(value)
        except (TypeError, ValueError):
            self.integer_value = 0
        try:
            self.double_value = float(value)
        except (TypeError, ValueError):
            self.double_value = 0.0


class _Parameter:
    def __init__(self, value):
        self.value = value

    def get_parameter_value(self):
        return _ParameterValue(self.value)


class _Logger:
    @staticmethod
    def info(message):
        rospy.loginfo("%s", message)

    @staticmethod
    def warn(message):
        rospy.logwarn("%s", message)

    @staticmethod
    def error(message):
        rospy.logerr("%s", message)


class _Now:
    @property
    def nanoseconds(self):
        return rospy.Time.now().to_nsec()

    @staticmethod
    def to_msg():
        return rospy.Time.now()


class _Clock:
    @staticmethod
    def now():
        return _Now()


class _Publisher:
    def __init__(self, publisher):
        self._publisher = publisher

    def publish(self, message):
        self._publisher.publish(message)

    def get_subscription_count(self):
        return self._publisher.get_num_connections()


class Node:
    """Subset of the rclpy Node API required by panel_detect_node."""

    def __init__(self, name):
        self._node_name = name
        self._parameter_defaults = {}
        self._resources = []
        self._destroyed = False
        self._callback_lock = threading.RLock()
        self.context = self

    def declare_parameter(self, name, default_value=None):
        self._parameter_defaults[name] = default_value
        return _Parameter(rospy.get_param("~" + name, default_value))

    def get_parameter(self, name):
        default = self._parameter_defaults.get(name)
        return _Parameter(rospy.get_param("~" + name, default))

    @staticmethod
    def get_logger():
        return _Logger()

    @staticmethod
    def get_clock():
        return _Clock()

    @staticmethod
    def create_publisher(message_type, topic, queue_size):
        return _Publisher(
            rospy.Publisher(topic, message_type, queue_size=int(queue_size)))

    def create_subscription(self, message_type, topic, callback, qos_profile):
        def serialized(message):
            with self._callback_lock:
                if not self._destroyed:
                    callback(message)
        subscription = rospy.Subscriber(
            topic, message_type, serialized, queue_size=int(qos_profile),
            buff_size=16 * 1024 * 1024)
        self._resources.append(subscription)
        return subscription

    def create_timer(self, period, callback):
        def serialized(event):
            with self._callback_lock:
                if not self._destroyed:
                    callback()
        timer = rospy.Timer(rospy.Duration(period), serialized)
        self._resources.append(timer)
        return timer

    @staticmethod
    def ok():
        return not rospy.is_shutdown()

    def destroy_node(self):
        with self._callback_lock:
            self._destroyed = True
        for resource in self._resources:
            if hasattr(resource, 'shutdown'):
                resource.shutdown()
            elif hasattr(resource, 'unregister'):
                resource.unregister()
        self._resources.clear()
