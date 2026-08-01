#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_node: 指令执行节点。
订阅 /command (std_msgs/String) — JSON 指令数组。
发布 /cmd_vel (geometry_msgs/Twist) — 底盘运动控制。
发布 /arm/grasp_command (std_msgs/String) — 机械臂抓取目标，由 arm_task_manager 消费。
发布 /emergency/abort (std_msgs/String) — 中止紧急呼叫，由紧急呼叫模块消费。
"""
import json
import math
import queue
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from voice_brain_module.commands import is_emergency_abort


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._arm_grasp_pub = self.create_publisher(String, "/arm/grasp_command", 10)
        self._emg_abort_pub = self.create_publisher(String, "/emergency/abort", 10)
        self.create_subscription(String, "/command", self._on_command, 10)

        self._work_q = queue.Queue()
        self._stop_requested = threading.Event()
        threading.Thread(target=self._work_loop, daemon=True).start()

        self.get_logger().info("control_node 就绪，等待 /command")

    def _on_command(self, msg: String):
        """紧急中止走快车道，其余入队。

        _work_loop 是串行的，一条"前进 10 米"能占着队列 50 秒。中止紧急呼叫
        排在它后面等，等于没做——所以在回调线程里当场转发（只是一次
        JSON 解析加一次 publish，微秒级，不会拖慢 ROS 执行器）。
        """
        rest = self._route_emergency(msg.data)
        if rest is None:
            self._work_q.put(msg.data)   # 解析失败：原样入队，由 _work_loop 统一报错
        elif rest:
            self._work_q.put(json.dumps(rest, ensure_ascii=False))

    def _route_emergency(self, cmd_json: str):
        """挑出紧急中止指令立即转发，返回剩余指令列表；解析失败返回 None。"""
        try:
            commands = json.loads(cmd_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(commands, list):
            return None

        aborts = [c for c in commands if is_emergency_abort(c)]
        if not aborts:
            return commands

        for cmd in aborts:
            params = cmd.get("params", {}) if isinstance(cmd.get("params"), dict) else {}
            payload = json.dumps({
                "event":     "emergency_abort",
                "action":    cmd.get("action", ""),
                "reason":    params.get("reason", "user_cancel"),
                "detector":  params.get("detector", ""),
                "utterance": params.get("utterance", ""),
                "stamp_sec": time.time(),
            }, ensure_ascii=False)
            self.get_logger().warn(f"🚨 中止紧急呼叫: {payload}")
            self._emg_abort_pub.publish(String(data=payload))

        return [c for c in commands if not is_emergency_abort(c)]

    def _work_loop(self):
        while rclpy.ok():
            cmd_json = self._work_q.get()
            self._stop_requested.clear()
            try:
                commands = json.loads(cmd_json)
                if not isinstance(commands, list):
                    self.get_logger().error("指令不是 JSON 数组")
                    continue
                for i, cmd in enumerate(commands, start=1):
                    if self._stop_requested.is_set():
                        self.get_logger().info("指令执行被中断")
                        break
                    self._execute_one(cmd, i)
            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON 解析失败: {e}")
            except Exception as e:
                self.get_logger().error(f"执行异常: {e}")
            finally:
                self._stop()

    def _execute_one(self, cmd: dict, idx: int):
        actuator = cmd.get("actuator", "")
        action   = cmd.get("action", "")
        params   = cmd.get("params", {})

        if actuator == "底盘":
            self._execute_chassis(action, params, idx)
        elif actuator == "机械臂":
            self._execute_arm(action, params, idx)
        else:
            self.get_logger().warn(f"第 {idx} 条: 未知执行机构 \"{actuator}\"")

    def _execute_chassis(self, action: str, params: dict, idx: int):
        twist = Twist()

        if action == "前进":
            speed    = params.get("speed", 0.2)
            distance = params.get("distance", 1.0)
            twist.linear.x = speed
            duration = distance / speed
            self.get_logger().info(f"▶ 前进 {distance:.1f}m ({speed:.2f} m/s, {duration:.1f}s)")
            self._publish_and_sleep(twist, duration)

        elif action == "后退":
            speed    = params.get("speed", 0.15)
            distance = params.get("distance", 0.5)
            twist.linear.x = -speed
            duration = distance / speed
            self.get_logger().info(f"◀ 后退 {distance:.1f}m ({speed:.2f} m/s, {duration:.1f}s)")
            self._publish_and_sleep(twist, duration)

        elif action == "左转":
            angle = params.get("angle", 90.0)
            speed = params.get("speed", 0.5)
            twist.angular.z = speed  # 正 = 左转
            duration = math.radians(angle) / speed
            self.get_logger().info(f"↺ 左转 {angle:.0f}° ({speed:.2f} rad/s, {duration:.1f}s)")
            self._publish_and_sleep(twist, duration)

        elif action == "右转":
            angle = params.get("angle", 90.0)
            speed = params.get("speed", 0.5)
            twist.angular.z = -speed  # 负 = 右转
            duration = math.radians(angle) / speed
            self.get_logger().info(f"↻ 右转 {angle:.0f}° ({speed:.2f} rad/s, {duration:.1f}s)")
            self._publish_and_sleep(twist, duration)

        elif action == "停止":
            self.get_logger().info("■ 停止")
            self._stop()

        else:
            pass  # 非本节点负责的动作，静默跳过

    def _execute_arm(self, action: str, params: dict, idx: int):
        """机械臂控制：发布抓取命令到 /arm/grasp_command，由 arm_task_manager 执行。"""
        if action == "抓取":
            target = params.get("target", "")
            if not target:
                self.get_logger().warn(f"第 {idx} 条: 机械臂抓取缺少 target 参数")
                return
            self.get_logger().info(f"✋ 机械臂抓取: {target}")
            self._arm_grasp_pub.publish(String(data=target))
        else:
            self.get_logger().warn(f"第 {idx} 条: 未知机械臂动作 \"{action}\"")

    def _publish_and_sleep(self, twist: Twist, duration: float):
        """发布 Twist，保持 duration 秒后停止。每 0.1s 检查打断信号。"""
        self._cmd_vel_pub.publish(twist)
        elapsed = 0.0
        while elapsed < duration and not self._stop_requested.is_set():
            time.sleep(0.1)
            elapsed += 0.1
        if not self._stop_requested.is_set():
            self._stop()

    def _stop(self):
        self._cmd_vel_pub.publish(Twist())


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
