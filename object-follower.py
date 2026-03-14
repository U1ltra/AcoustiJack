import asyncio
import argparse
from mavsdk import System
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from simple_pid import PID
import time


class ControlSubscriber(Node):
    def __init__(self, drone, height=10.0, stable=False):
        super().__init__("control_subscriber")

        self.subscription = self.create_subscription(
            Point, "object/follow", self.listener_callback, 10
        )
        self.ready_pub = self.create_publisher(Bool, "/start_movement", 10)

        self.drone = drone

        # PID Controllers for drone movement.
        # P = proportional (current), I = integral (accumulate), D = derivative (dampen)
        self.pitch_pid = PID(0.005, 0.0, 0.0, setpoint=0, output_limits=(-0.5, 0.5))
        self.roll_pid = PID(
            0.001, 0.0, 0.0, setpoint=0, output_limits=(-0.5, 0.5)
        )
        self.throttle_pid = PID(
            0.0,
            0.0,
            0.0,
            setpoint=10,
            output_limits=(0, 1),
            starting_output=0.5,
        )
        self.yaw_pid = PID(
            -0.005, -0.0, -0.001, setpoint=0, output_limits=(-0.5, 0.5)
        )

        self.control_data = [0.0, 0.0, 0.0, 0.0]
        self.height = height
        self.stable = stable

    async def control_loop(self):
        start = time.time()
        print("Waiting to connect...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break

        print("-- Arming")
        await self.drone.action.arm()
        print("-- Armed!")

        print("-- Taking off")
        await self.drone.action.set_takeoff_altitude(self.height)
        await self.drone.action.takeoff()

        print("-- Waiting to reach target altitude")
        async for position in self.drone.telemetry.position():
            if position.relative_altitude_m >= self.height * 0.95:
                print(
                    f"-- Reached target altitude: {position.relative_altitude_m:.1f}m"
                )
                self.throttle_pid.setpoint = position.relative_altitude_m
                break
        print(f"-- Control loop started at {time.time() - start:.2f} seconds")

        await self.drone.manual_control.set_manual_control_input(
            self.control_data[0],
            self.control_data[1],
            self.control_data[2],
            self.control_data[3],
        )
        print("-- Ready to receive control data")

        while True:
            # print('Control: ', self.control_data)
            await self.drone.manual_control.set_manual_control_input(
                self.control_data[0],
                self.control_data[1],
                self.control_data[2],
                self.control_data[3],
            )
            await asyncio.sleep(0.02)

    def listener_callback(self, data):
        # print('Error: ', data)
        # return
        if not self.stable:
            self.control_data[0] = self.pitch_pid(data.y)
            self.control_data[1] = self.roll_pid(data.x)
            self.control_data[2] = self.throttle_pid(data.z)
            self.control_data[3] = self.yaw_pid(data.x)
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Control subscriber")
    parser.add_argument(
        "--height", type=float, help="Height of the drone", default=10.0
    )
    parser.add_argument(
        "--stable", action="store_true", help="Stable control", default=False
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    rclpy.init()

    drone = System()
    control_subscriber = ControlSubscriber(drone, args.height, args.stable)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(control_subscriber)

    try:
        loop = asyncio.get_event_loop()

        await asyncio.gather(
            control_subscriber.control_loop(), loop.run_in_executor(None, executor.spin)
        )

    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
