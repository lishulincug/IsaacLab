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
from zerith_h1_shadow_cfg import (  # noqa: E402
    ARM_JOINTS,
    DEFAULT_TOPICS,
    LEFT_ARM_JOINTS,
    REAL_JOINT_STATE_INDICES,
    RIGHT_ARM_JOINTS,
    SIM_TOPICS,
    ShadowTopics,
    decode_arm_command,
    decode_real_joint_state,
    encode_sim_joint_state,
    validate_standalone_topics,
)

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
def test_standalone_feedback_uses_pico_21_element_layout() -> None:
    values = encode_sim_joint_state(np.arange(14, dtype=np.float64))
    np.testing.assert_array_equal(values[5:12], np.arange(7))
    np.testing.assert_array_equal(values[14:21], np.arange(7, 14))
    np.testing.assert_array_equal(values[[0, 1, 2, 3, 4, 12, 13]], np.zeros(7))


def test_standalone_rejects_real_robot_topic_names() -> None:
    validate_standalone_topics(SIM_TOPICS)
    try:
        validate_standalone_topics(ShadowTopics())
    except ValueError:
        pass
    else:
        raise AssertionError("Standalone mode accepted real robot topics")


def test_shadow_runner_publishes_only_inside_the_standalone_branch() -> None:
    tree = ast.parse((TELEOP_DIR / "zerith_h1_shadow.py").read_text(encoding="utf-8"))
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "create_publisher" in calls and "publish" in calls
    source = (TELEOP_DIR / "zerith_h1_shadow.py").read_text(encoding="utf-8")
    assert 'if mode == "standalone"' in source
    assert 'if args_cli.mode == "standalone"' in source
    assert tuple(ARM_JOINTS[:7]) == LEFT_ARM_JOINTS
