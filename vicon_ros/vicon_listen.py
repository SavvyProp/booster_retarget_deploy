# utils/vicon_tf_client.py
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
import numpy as np


class ViconTFClient:
    def __init__(self,
                 namespace: str = 'vicon',
                 world_frame: str = 'world'):
        # Ensure rclpy is initialized before creating an executor/node
        if not rclpy.ok():
            rclpy.init(args=None)
        self._exec = rclpy.executors.SingleThreadedExecutor()
        self.node            = Node('vicon_tf_client')
        self._exec.add_node(self.node)
        self.tf_buffer       = Buffer()
        self.tf_listener     = TransformListener(self.tf_buffer, self.node)
        self.ns              = namespace
        self.full_world      = f'{self.ns}/{world_frame}'
        self._last_ball_pos  = None

    @staticmethod
    def _quat_2_rpy(q):
        x, y, z, w = q
        
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp =   2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        return rpy

    def spin_once(self, timeout: float = 0.001):
        """Process incoming TF messages so buffer stays up to date."""
        self._exec.spin_once(timeout_sec=timeout)

    def get_marker_position(self, segment: str, timeout: float = 0.001) -> tuple[float, float, float]:
        """
        Looks up `ns/world` → `ns/{segment}` and returns (x,y,z).
        Raises a TransformException if lookup fails.
        """
        self.spin_once(timeout)

        target = f'{self.ns}/{segment}'
        ts = self.tf_buffer.lookup_transform(
            self.full_world,
            target,
            Time()
        )
        t = ts.transform.translation
        r = ts.transform.rotation
        
        return np.array([t.x, t.y, t.z]), np.array([r.w, r.x, r.y, r.z])
    
    def get_marker_position_quat(self, segment: str, timeout: float = 0.001) -> tuple[float, float, float]:
        """
        Looks up `ns/world` → `ns/{segment}` and returns (x,y,z).
        Raises a TransformException if lookup fails.
        """
        self.spin_once(timeout)

        target = f'{self.ns}/{segment}'
        ts = self.tf_buffer.lookup_transform(
            self.full_world,
            target,
            Time()
        )
        t = ts.transform.translation
        r = ts.transform.rotation
        
        return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])
    
    def get_marker_position_unlabled(self, idx: int, timeout: float = 0.001) -> tuple[float, float, float]:
        """
        Looks up `ns/world` → `ns/{segment}` and returns (x,y,z).
        Raises a TransformException if lookup fails.
        """
        self.spin_once(timeout)

        target = f'{self.ns}/unlabeled_{idx}'
        ts = self.tf_buffer.lookup_transform(
            self.full_world,
            target,
            Time()
        )
        t = ts.transform.translation
        return np.array([t.x, t.y, t.z])
    
    
    def print_all_frames(self, timeout: float = 0.001) -> None:
        """
        Spin TF once, then print every frame under `ns/*` along with its [x,y,z].
        """
        # 1) pump callbacks so buffer is up to date
        self.spin_once(timeout)

        # 2) get raw frame list (bytes on disk)
        frames = self.tf_buffer._getFrameStrings()
        print(frames)

        for raw in frames:
            # 3) decode to str
            child = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw
            # 4) only Vicon frames
            if not child.startswith(self.ns + '/'):
                continue

            try:
                # 5) lookup world→child
                ts = self.tf_buffer.lookup_transform(
                    self.full_world,
                    child,
                    Time()
                )
                t = ts.transform.translation
                # 6) print name + position
                print(f"{child}: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")

            except Exception as e:
                print(f"  (failed to lookup {child}: {e})")

    def get_ball_pos(self, timeout: float = 0.001) -> np.ndarray | None:
        try:
            ball_pos, _ = self.get_marker_position("Ball/Ball")
        except:
            if self._last_ball_pos is None:
                return np.array([0, 0, 0])
            return self._last_ball_pos
        return ball_pos
    
