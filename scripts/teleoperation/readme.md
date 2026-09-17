纯 IsaacLab 遥操请使用闭环配置：

# 终端 1：IsaacLab
cd D:\robot\sim\IsaacLab
.\isaaclab.bat -p scripts\teleoperation\zerith_h1_shadow.py --mode standalone

# 终端 2：Pico；必须 use_sim:=false 才会消费 IsaacLab 回传的关节状态
ros2 run pico_teleop teleop_h1_zerith --ros-args `
  -p standalone_mode:=true `
  -p topic_prefix:=/sim `
  -p use_sim:=false

数据链路是：

Pico CasADi IK
  → /sim/control/mux/in/xr/*_arm/motors_cmd
  → IsaacLab
  → /sim/h1/joint_states
  → Pico 反馈闭环