# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Isaac Lab shadow and isolated standalone teleoperation for Zerith H1.

Launch with ``./isaaclab.bat -p scripts/teleoperation/zerith_h1_shadow.py``.
Shadow mode creates no ROS publisher. Standalone mode publishes only /sim/h1/joint_states.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from isaaclab.app import AppLauncher
from zerith_h1_shadow_cfg import (ARM_JOINTS, DEFAULT_COMMAND_TIMEOUT_S, DEFAULT_LOG_DIR, DEFAULT_PHYSICS_DT_S, DEFAULT_TOPICS, SIM_TOPICS, ShadowTopics, ZERITH_H1_USD_PATH, decode_arm_command, decode_real_joint_state, encode_sim_joint_state, validate_standalone_topics)

parser = argparse.ArgumentParser(description="Read-only Zerith H1 Isaac Lab shadow runner.")
parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
parser.add_argument("--command_timeout_s", type=float, default=DEFAULT_COMMAND_TIMEOUT_S)
parser.add_argument("--physics_dt", type=float, default=DEFAULT_PHYSICS_DT_S)
parser.add_argument("--record_mux_inputs", action="store_true")
parser.add_argument("--record_xr", action="store_true")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--mode", choices=("shadow", "standalone"), default="shadow")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

@dataclass
class TimedVector:
    value: np.ndarray | None = None
    received_s: float = 0.0
    ros_stamp_ns: int | None = None
    source: str = "missing"

def stamp_ns(message: Any) -> int | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    return None if stamp is None else int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

class ShadowRosSubscriber:
    """ROS bridge whose only publishing capability is isolated standalone feedback."""
    def __init__(self, topics: ShadowTopics, record_mux_inputs: bool, record_xr: bool, mode: str):
        import rclpy
        from sensor_msgs.msg import JointState
        self._rclpy, self._owns_context, self.mode = rclpy, not rclpy.ok(), mode
        if mode == "standalone":
            validate_standalone_topics(topics)
        if self._owns_context:
            rclpy.init()
        self.node = rclpy.create_node("zerith_h1_isaaclab_shadow")
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.lock = Lock()
        self.final_left, self.final_right, self.real = TimedVector(), TimedVector(), TimedVector()
        self.mux_left, self.mux_right = TimedVector(), TimedVector()
        self.xr_receipts: list[tuple[float, int | None]] = []
        self.subscriptions = [
            self.node.create_subscription(JointState, topics.final_left_arm, self._on_final_left, 20),
            self.node.create_subscription(JointState, topics.final_right_arm, self._on_final_right, 20),
        ]
        self.joint_state_publisher = None
        if mode == "shadow":
            self.subscriptions.append(self.node.create_subscription(JointState, topics.joint_states, self._on_real, 50))
        else:
            self.joint_state_publisher = self.node.create_publisher(JointState, topics.joint_states, 50)
        if record_mux_inputs:
            self.subscriptions += [
                self.node.create_subscription(JointState, topics.mux_input_left_arm, self._on_mux_left, 20),
                self.node.create_subscription(JointState, topics.mux_input_right_arm, self._on_mux_right, 20),
            ]
        if record_xr:
            try:
                from picoxr.msg import Custom
            except ImportError as error:
                raise RuntimeError("--record_xr requires the picoxr ROS 2 message package.") from error
            self.subscriptions.append(self.node.create_subscription(Custom, topics.xr_pose, self._on_xr, 50))
    def _store(self, target: TimedVector, values: np.ndarray | None, source: str, message: Any) -> None:
        if values is None:
            self.node.get_logger().warning("Ignored malformed JointState: %s", source)
            return
        with self.lock:
            target.value, target.received_s, target.ros_stamp_ns, target.source = values, time.monotonic(), stamp_ns(message), source
    def _on_final_left(self, message: Any) -> None:
        values, source = decode_arm_command(message, ARM_JOINTS[:7], "left"); self._store(self.final_left, values, source, message)
    def _on_final_right(self, message: Any) -> None:
        values, source = decode_arm_command(message, ARM_JOINTS[7:], "right"); self._store(self.final_right, values, source, message)
    def _on_real(self, message: Any) -> None:
        values, source = decode_real_joint_state(message); self._store(self.real, values, source, message)
    def _on_mux_left(self, message: Any) -> None:
        values, source = decode_arm_command(message, ARM_JOINTS[:7], "left"); self._store(self.mux_left, values, source, message)
    def _on_mux_right(self, message: Any) -> None:
        values, source = decode_arm_command(message, ARM_JOINTS[7:], "right"); self._store(self.mux_right, values, source, message)
    def _on_xr(self, message: Any) -> None:
        with self.lock: self.xr_receipts.append((time.monotonic(), getattr(message, "timestamp_ns", None)))
    def spin_once(self) -> None:
        self.executor.spin_once(timeout_sec=0.0)
    def publish_sim_joint_state(self, arm_positions: np.ndarray) -> None:
        """Publish simulated feedback only when standalone namespace validation passed."""
        if self.joint_state_publisher is None:
            return
        from sensor_msgs.msg import JointState
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = [f"sim_joint_{index}" for index in range(21)]
        message.position = encode_sim_joint_state(arm_positions).tolist()
        self.joint_state_publisher.publish(message)
    def snapshot(self) -> tuple[TimedVector, TimedVector, TimedVector, TimedVector, TimedVector]:
        with self.lock:
            return tuple(TimedVector(**item.__dict__) for item in (self.final_left, self.final_right, self.real, self.mux_left, self.mux_right))
    def xr_receipt_count(self) -> int:
        with self.lock:
            return len(self.xr_receipts)
    def close(self) -> None:
        self.executor.remove_node(self.node); self.node.destroy_node()
        if self._owns_context: self._rclpy.shutdown()

class ShadowLogger:
    def __init__(self, root: Path):
        self.dir = root / time.strftime("%Y%m%d_%H%M%S"); self.dir.mkdir(parents=True, exist_ok=False)
        self.file = (self.dir / "samples.csv").open("w", newline="", encoding="utf-8")
        self.fields = ["monotonic_s", "final_age_s", "real_age_s", "fresh", "aligned", "timeout_count", "xr_receipt_count", "final_ros_stamp_ns", "real_ros_stamp_ns", "feedback_minus_command_ros_s", "command_source", "feedback_source"] + [f"{prefix}_{joint}" for prefix in ("mux_input", "target", "real", "sim", "real_error", "sim_error", "sim_minus_real") for joint in ARM_JOINTS]
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields); self.writer.writeheader()
        self.real_sq: list[np.ndarray] = []; self.sim_sq: list[np.ndarray] = []; self.mux_delta_sq: list[np.ndarray] = []; self.feedback_lag_s: list[float] = []
        self.real_peak = np.zeros(14); self.sim_peak = np.zeros(14)
    def write(self, now: float, mux_input: np.ndarray | None, target: np.ndarray | None, real: np.ndarray | None, sim: np.ndarray, final: TimedVector, feedback: TimedVector, aligned: bool, timeout_count: int, xr_receipt_count: int) -> None:
        feedback_lag_s = None if final.ros_stamp_ns is None or feedback.ros_stamp_ns is None else (feedback.ros_stamp_ns-final.ros_stamp_ns)/1_000_000_000
        row: dict[str, Any] = {"monotonic_s": now, "final_age_s": now-final.received_s if final.value is not None else math.inf, "real_age_s": now-feedback.received_s if feedback.value is not None else math.inf, "fresh": target is not None, "aligned": aligned, "timeout_count": timeout_count, "xr_receipt_count": xr_receipt_count, "final_ros_stamp_ns": final.ros_stamp_ns, "real_ros_stamp_ns": feedback.ros_stamp_ns, "feedback_minus_command_ros_s": feedback_lag_s, "command_source": final.source, "feedback_source": feedback.source}
        for prefix, values in (("mux_input", mux_input), ("target", target), ("real", real), ("sim", sim)):
            for joint, value in zip(ARM_JOINTS, values if values is not None else np.full(14, np.nan), strict=True): row[f"{prefix}_{joint}"] = float(value)
        if target is not None and real is not None:
            re, se, sr = real-target, sim-target, sim-real
            self.real_sq.append(re**2); self.sim_sq.append(se**2); self.real_peak=np.maximum(self.real_peak, np.abs(re)); self.sim_peak=np.maximum(self.sim_peak, np.abs(se))
        else: re = se = sr = np.full(14, np.nan)
        if mux_input is not None and target is not None:
            self.mux_delta_sq.append((mux_input-target)**2)
        if feedback_lag_s is not None:
            self.feedback_lag_s.append(feedback_lag_s)
        for prefix, values in (("real_error", re), ("sim_error", se), ("sim_minus_real", sr)):
            for joint, value in zip(ARM_JOINTS, values, strict=True): row[f"{prefix}_{joint}"] = float(value)
        self.writer.writerow(row)
    def close(self) -> None:
        self.file.close(); summary: dict[str, Any] = {"joint_names": list(ARM_JOINTS), "samples_with_complete_comparison": len(self.real_sq)}
        if self.real_sq:
            summary.update(real_tracking_rms_rad=np.sqrt(np.mean(self.real_sq, axis=0)).tolist(), sim_tracking_rms_rad=np.sqrt(np.mean(self.sim_sq, axis=0)).tolist(), real_tracking_peak_rad=self.real_peak.tolist(), sim_tracking_peak_rad=self.sim_peak.tolist())
        if self.mux_delta_sq:
            summary["mux_input_to_final_rms_rad"] = np.sqrt(np.mean(self.mux_delta_sq, axis=0)).tolist()
        if self.feedback_lag_s:
            summary["feedback_minus_command_ros_s"] = {"median": float(np.median(self.feedback_lag_s)), "mean": float(np.mean(self.feedback_lag_s))}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def robot_cfg() -> ArticulationCfg:
    if not ZERITH_H1_USD_PATH.exists(): raise FileNotFoundError(ZERITH_H1_USD_PATH)
    return ArticulationCfg(prim_path="/World/ZerithH1", spawn=sim_utils.UsdFileCfg(usd_path=str(ZERITH_H1_USD_PATH), variants={"Physics":"PhysX", "Robot":"Robot", "Sensor":"None"}, articulation_props=sim_utils.ArticulationRootBaseCfg(fix_root_link=True)))

def main() -> None:
    if args_cli.command_timeout_s <= 0 or args_cli.physics_dt <= 0: raise ValueError("Timeout and physics dt must be positive.")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)); sim.set_camera_view([3.0,-3.0,2.0], [0.0,0.0,1.1])
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75)); light.func("/World/Light", light)
    robot = Articulation(robot_cfg()); sim.reset()
    ids, names = robot.find_joints(list(ARM_JOINTS), preserve_order=True)
    if tuple(names) != ARM_JOINTS: raise RuntimeError(f"USD joint mapping mismatch: {names}")
    ids = torch.tensor(ids, dtype=torch.long, device=sim.device)
    topics = DEFAULT_TOPICS if args_cli.mode == "shadow" else SIM_TOPICS
    ros = ShadowRosSubscriber(topics, args_cli.record_mux_inputs, args_cli.record_xr, args_cli.mode)
    logger = ShadowLogger(args_cli.log_dir)
    aligned = False; timeout_count = 0; step = 0
    try:
        while simulation_app.is_running() and (args_cli.max_steps == 0 or step < args_cli.max_steps):
            ros.spin_once(); now = time.monotonic(); left, right, feedback, mux_left, mux_right = ros.snapshot()
            fresh = left.value is not None and right.value is not None and now-left.received_s <= args_cli.command_timeout_s and now-right.received_s <= args_cli.command_timeout_s
            target = np.concatenate((left.value, right.value)) if fresh else None
            mux_input = np.concatenate((mux_left.value, mux_right.value)) if mux_left.value is not None and mux_right.value is not None else None
            if target is None: timeout_count += 1
            initial_state = feedback.value if args_cli.mode == "shadow" else target
            if not aligned and initial_state is not None:
                q = torch.as_tensor(initial_state, dtype=torch.float32, device=sim.device).unsqueeze(0)
                robot.write_joint_position_to_sim_index(position=q, joint_ids=ids); robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids); robot.reset(); aligned = True
            if aligned:
                hold_or_target = target
                if hold_or_target is None:
                    hold_or_target = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy()
                robot.set_joint_position_target_index(target=torch.as_tensor(hold_or_target, dtype=torch.float32, device=sim.device).unsqueeze(0), joint_ids=ids)
            robot.write_data_to_sim(); sim.step(); robot.update(args_cli.physics_dt)
            sim_arm_positions = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy()
            if args_cli.mode == "standalone":
                ros.publish_sim_joint_state(sim_arm_positions)
            logger.write(now, mux_input, target, feedback.value, sim_arm_positions, left, feedback, aligned, timeout_count, ros.xr_receipt_count()); step += 1
    finally:
        logger.close(); ros.close()

if __name__ == "__main__":
    try: main()
    finally: simulation_app.close()
