# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
TELEOP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP_DIR))
from zerith_h1_shadow_cfg import ARM_JOINTS, LEFT_ARM_JOINTS, REAL_JOINT_STATE_INDICES, RIGHT_ARM_JOINTS, decode_arm_command, decode_real_joint_state  # noqa: E402

@dataclass
class FakeJointState:
    name: list[str]
    position: list[float]

def test_named_commands_are_reordered() -> None:
    values, source = decode_arm_command(FakeJointState(list(reversed(LEFT_ARM_JOINTS)), list(range(7))), LEFT_ARM_JOINTS, "left")
    assert source == "named"; np.testing.assert_array_equal(values, np.arange(6, -1, -1))
def test_generic_pico_commands_use_positional_order() -> None:
    values, source = decode_arm_command(FakeJointState([f"right_arm_joint_{i}" for i in range(7)], list(range(7))), RIGHT_ARM_JOINTS, "right")
    assert source == "positional"; np.testing.assert_array_equal(values, np.arange(7))
def test_partial_commands_and_non_finite_values_are_rejected() -> None:
    assert decode_arm_command(FakeJointState(list(LEFT_ARM_JOINTS[:-1]), list(range(6))), LEFT_ARM_JOINTS, "left")[0] is None
    assert decode_arm_command(FakeJointState(list(RIGHT_ARM_JOINTS), [0.0]*6+[float("nan")]), RIGHT_ARM_JOINTS, "right")[0] is None
def test_unnamed_feedback_uses_existing_controller_layout() -> None:
    values, source = decode_real_joint_state(FakeJointState([], list(range(21))))
    assert source == "controller_layout"; np.testing.assert_array_equal(values, np.asarray(list(range(21)))[list(REAL_JOINT_STATE_INDICES)])
def test_shadow_runner_never_creates_a_ros_publisher() -> None:
    tree = ast.parse((TELEOP_DIR / "zerith_h1_shadow.py").read_text(encoding="utf-8"))
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "create_publisher" not in calls and "publish" not in calls
    assert tuple(ARM_JOINTS[:7]) == LEFT_ARM_JOINTS
