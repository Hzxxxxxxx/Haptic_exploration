#!/usr/bin/env python3
"""
读取 q_des_log.csv 并绘制 7 个关节角随时间变化曲线
"""
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

csv_path = pathlib.Path("/home/mscrobotics2425laptop16/mujoco_ros_ws/q_des_log.csv")
if not csv_path.exists():
    sys.exit("❌ 未找到 q_des_log.csv，请先运行控制器节点产生日志")

# 1) 读取数据
df = pd.read_csv(csv_path)
time = df["time"]

# 2) 绘图
plt.figure(figsize=(10, 6))
for j in range(1, 8):  # q1 … q7
    plt.plot(time, df[f"q{j}"], label=f"q{j}")

plt.xlabel("Time [s]")
plt.ylabel("Joint angle [rad]")
plt.title("Desired Joint Positions (q_des) vs. Time")
plt.legend(ncol=4, fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()

# 如需保存图片，取消下行注释：
# plt.savefig("q_des_curves.png", dpi=150)
