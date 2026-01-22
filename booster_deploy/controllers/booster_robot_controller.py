from __future__ import annotations
import logging
import signal
import time
import threading
import multiprocessing as mp
from multiprocessing import synchronize

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from booster_interface.msg import LowState, LowCmd, MotorCmd

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    RobotMode,
)

from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.synced_array import SyncedArray
from ..utils.metrics import SyncedMetrics
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import RemoteControlService
from ..utils.math import rotate_vector_by_quat
from vicon_ros.vicon_listen import ViconTFClient

logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class CountTimer:
    def __init__(self, dt: float = 0.002, use_sim_time: bool = False):
        self.dt = dt
        # Use multiprocessing.Value for inter-process communication
        self.counter = mp.Value('L', 0)
        self.use_sim_time = use_sim_time

    def tick_timer_if_sim(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                self.counter.value += 1

    def get_time(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                return self.counter.value * self.dt
        else:
            return time.perf_counter()


class BoosterRobotPortal:
    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(self, cfg: ControllerCfg, use_sim_time: bool = False) -> None:
        self.cfg = cfg
        self.robot = BoosterRobot(cfg.robot)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.remoteControlService = RemoteControlService()
        # Use multiprocessing.Event for inter-process communication
        self.exit_event = mp.Event()
        self.is_running = True
        self.timer = CountTimer(
            self.cfg.booster.low_state_dt, use_sim_time=use_sim_time)

        def signal_handler(sig, frame):
            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt received. Shutting down...")
            self.exit_event.set()

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self.inference_process = None  # Inference process reference
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.low_state_thread = None
        self.low_cmd_process: mp.Process | None = None

        rclpy.init()

        self.vicon_pos = np.zeros(3, dtype=np.float32)
        self.vicon_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_time = time.time()
        self.global_vel = np.zeros(3, dtype=np.float32)
        self.local_vel = np.zeros(3, dtype=np.float32)
        self.global_pos = np.zeros(3, dtype=np.float32)
        self.global_ori = np.zeros(9, dtype=np.float32)

        

        self.vicon_client = ViconTFClient()
        # Initialize communication. Callbacks may start immediately and
        # reference `is_running` and `exit_event`, so ensure those are set.
        self._init_communication()

    def _init_synced_buffer(self):
        action_dtype = np.dtype(
            [
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.ndarray((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("root_rpy_w", float, (3,)),
                ("root_mat_w", float, (9,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_b", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = SyncedArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )

    def _init_metrics(self):
        # initialize cross-process synced metrics
        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_communication(self) -> None:
        try:
            self.client = B1LocoClient()
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _start_low_state_subscription(self) -> None:
        """Start ROS 2 subscription loop on a dedicated thread.

        The subscription is run on a dedicated thread and spins a
        SingleThreadedExecutor for the `/low_state` topic.
        """

        def low_state_service_executor():
            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)

            try:
                # loop: check exit_event and rclpy.ok()
                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:
                # Suppress RCLError if we are shutting down
                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def get_velocity(self):
        try:
            vicon_pos, vicon_quat, rpy = self.vicon_client.get_marker_position(
                "Booster/booster_seg"
                )

            marker_offset_body = np.array([0.150, 0.0, 0.162]) # top-center of Booster
            R_meas_body = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0,-1.0, 0.0]])

            # Additional fixed pitch tilt of the marker plane by +6 deg about BODY Y (tilt defined in true body frame)
            theta = np.deg2rad(18.36)
            R_body_markers = np.array([[ np.cos(theta), 0.0, np.sin(theta)],
                                        [ 0.0,          1.0, 0.0         ],
                                        [-np.sin(theta), 0.0, np.cos(theta)]])

            # Total body->measured mapping including mounting tilt (apply body tilt first, then body->measured axis mapping)
            marker_offset_meas = R_meas_body @ (R_body_markers @ marker_offset_body)
            cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
            cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
            cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
            R_x = np.array([[1.0, 0.0, 0.0],
                                [0.0,  cr, -sr],
                                [0.0,  sr,  cr]])
            R_y = np.array([[ cp, 0.0, sp],
                                [0.0, 1.0, 0.0],
                                [-sp, 0.0, cp]])
            R_z = np.array([[ cy, -sy, 0.0],
                                [ sy,  cy, 0.0],
                                [0.0, 0.0, 1.0]])
            R_world_meas = R_z @ R_y @ R_x
            R_world_body = R_world_meas @ (R_meas_body @ R_body_markers)

            st = time.time()
            dt = st - self.last_time
            self.last_time = st
            alpha = 0.2
            raw_global_vel = (vicon_pos - self.vicon_pos) / dt
            self.vicon_pos = vicon_pos
            self.global_vel = self.global_vel * (1 - alpha) + raw_global_vel * alpha

            self.local_vel = np.linalg.inv(R_world_body) @ self.global_vel
            self.logger.info("Local vel x: {:.3f} y: {:.3f} z: {:.3f}".format(
                self.local_vel[0], self.local_vel[1], self.local_vel[2]))
            self.global_pos = vicon_pos
            self.global_ori = R_world_body.flatten()
        except Exception as e:
            print("Failed to get marker position:", e)

    def _low_state_handler(self, low_state_msg: LowState):
        self.metrics["low_state_handler"].mark()
        try:
            if not self.is_running or self.exit_event.is_set():
                return

            # simulator tick
            self.timer.tick_timer_if_sim()

            # collect state data
            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            
            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["root_mat_w"][:] = self.global_ori
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = self.global_pos
            self._state_buf[0]["root_lin_vel_b"][:] = self.local_vel
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            
            self.synced_state.write(self._state_buf)

            # update velocity commands to synced_command
            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
            cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
            cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            self.synced_command.write(cmd)

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.running = False
            self.exit_event.set()

    def create_low_cmd_publisher(self, name):
        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        # construct low_cmd struct
        self.low_cmd = LowCmd()  # type: ignore
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL   # type: ignore
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]  # type: ignore
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        return publisher

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)

        self.logger.info("Subscriber found, starting control loop")        

        prepare_state = self.robot.cfg.prepare_state
        init_joint_pos = self.synced_state.read()[0]['joint_pos']
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = init_joint_pos[i]
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])

        self.low_cmd_publisher.publish(self.low_cmd)
        time.sleep(0.1)

        # change to custom mode
        self.client.ChangeMode(RobotMode.kCustom)
        # for i in range(20):  # try multiple times to make sure mode is changed
        #     self.client.ChangeMode(RobotMode.kCustom)
        #     time.sleep(0.5)
        #     if (mode:= self.client.GetStatus().current_mode) == RobotMode.kCustom:
        #         break
        # else:
        #     self.logger.error("Failed to switch to custom mode")
        #     return False

        trans = np.linspace(init_joint_pos, prepare_state.joint_pos, num=500)
        start_time = self.timer.get_time()
        for i in range(500):
            for j in range(self.robot.num_joints):
                self.motor_cmd[j].q = trans[i][j]
            self.low_cmd_publisher.publish(self.low_cmd)
            while self.timer.get_time() < start_time + (i + 1) * 0.002:
                time.sleep(0.0002)
        self.logger.info("Custom mode started, initialized with prepare pose")
        return True

    def start_rl_gait_conditionally(self):
        """Start RL gait and spawn inference process and publisher thread."""
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        # start inference process (separate process)
        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(
                self.cfg,
                self,
            ),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference process started")

        print(f"{self.remoteControlService.get_operation_hint()}")
        return True

    def cleanup(self) -> None:
        """Clean up resources (idempotent)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self.logger.info("Doing cleanup...")

        # stop threads and processes
        self.is_running = False
        self.exit_event.set()

        # wait for inference process
        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            self.logger.info("Waiting for inference process...")
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                self.logger.warning(
                    "Inference process did not stop, terminating...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        # close communications
        try:
            self.remoteControlService.close()
        except Exception as e:
            self.logger.error(f"Error closing remote control: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.low_state_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self.logger.error(f"Error waiting for low state thread: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Cleanup complete")

        # Print synced metrics summary to stdout
        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )

    def run(self):
        """Main loop: monitor inference process and diagnostics (10Hz)."""

        print("Initialization complete.")

        # start custom mode (interruptible)
        if not self.start_custom_mode_conditionally():
            print("Custom mode initialization cancelled.")
        # start RL gait (interruptible)
        elif not self.start_rl_gait_conditionally():
            print("RL gait initialization cancelled.")
        else:
            # main loop: wait for exit signal
            print("Inference Running.")
            while self.is_running and not self.exit_event.is_set():
                # check whether the inference process is alive
                if self.inference_process is not None:
                    inference_process_alive = self.inference_process.is_alive()
                    if not inference_process_alive:
                        self.logger.error("Inference process died unexpectedly")
                        self.is_running = False
                        self.exit_event.set()
                        break
                self.get_velocity()
                self.logger.info("Local vel x: {:.3f} y: {:.3f} z: {:.3f}".format(
                    self.local_vel[0], self.local_vel[1], self.local_vel[2]))
                time.sleep(0.001)
                #time.sleep(0.1)

        # exit and switch to walking mode
        self.logger.info("Exiting controller, switching to walking mode...")
        self.client.ChangeMode(RobotMode.kWalking)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
    ) -> None:
        BoosterRobotController(cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal
        slice_size = 3 * self.robot.num_joints + 7 + 12 + self.policy.obs_size
        self.obs_list = np.zeros((500, slice_size), dtype=np.float32)
        
        

    def update_vel_command(self):
        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = cmd["vx"] * self.vel_command.vx_max
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def update_state(self) -> None:
        state = self.portal.synced_state.read()[0]

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        #self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
        #    self.robot.data.root_quat_w,
        #    torch.from_numpy(
        #        state["root_lin_vel_w"]).to(dtype=torch.float32).to(
        #            self.robot.data.device)
        #)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            state["root_lin_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device
        )

        info_slice = self.robot_slice(
            state["root_pos_w"],
            state["root_mat_w"]
        )
        return info_slice

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        for i in range(self.robot.num_joints):
            self.portal.motor_cmd[i].q = float(dof_targets[i].item())
            kp_val = float(self.robot.joint_stiffness[i].item())
            kd_val = float(self.robot.joint_damping[i].item())
            self.portal.motor_cmd[i].kp = kp_val
            self.portal.motor_cmd[i].kd = kd_val
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)

    def stop(self):
        np.savetxt("eval_data/booster_obs_log.csv", self.obs_list, delimiter=",")
        super().stop()
        self.portal.exit_event.set()

    def robot_slice(self, global_pos, global_ori):
        joint_pos = self.robot.data.joint_pos.cpu().numpy()
        joint_vel = self.robot.data.joint_vel.cpu().numpy()
        root_lin_vel_b = self.robot.data.root_lin_vel_b.cpu().numpy()
        root_ang_vel_b = self.robot.data.root_ang_vel_b.cpu().numpy()
        time_stamp = np.array([self.portal.timer.get_time()])
        obs = self.policy.obs
        flat_obs = np.concatenate([
            time_stamp,
            joint_pos,
            joint_vel,
            root_lin_vel_b,
            root_ang_vel_b, 
            global_pos,
            global_ori.reshape(-1),
            obs
        ], axis = -1)
        return flat_obs
    
    def run(self):
        self.update_state()
        if self.vel_command is not None:
            self.update_vel_command()
        self.start()
        next_inference_time = self.portal.timer.get_time()
        last_save = time.time()

        while self.is_running and not self.portal.exit_event.is_set():
            if self.portal.timer.get_time() < next_inference_time:
                time.sleep(0.0002)
                continue
            if last_save + 1.0 < time.time():
                #np.savetxt("eval_data/booster_obs_log.csv", self.obs_list, delimiter=",")
                last_save = time.time()
            next_inference_time += self.cfg.policy_dt

            info_slice = self.update_state()
            self.portal.metrics["policy_step"].mark()
            dof_targets = self.policy_step()
            #info_slice = self.robot_slice(dof_targets)
            info_slice = np.concatenate([info_slice, dof_targets], axis = -1)
            self.obs_list = np.roll(self.obs_list, -1, axis=0)
            self.obs_list[-1, :] = info_slice
            #print("Dof targets:", dof_targets.cpu().numpy())
            self.ctrl_step(dof_targets)
        np.savetxt("eval_data/booster_obs_log.csv", self.obs_list, delimiter=",")
        self.portal.exit_event.set()