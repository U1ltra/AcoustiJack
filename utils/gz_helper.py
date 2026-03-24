import math
import time
from simple_pid import PID
import gz.transport13 as transport
from gz.msgs10 import world_control_pb2, boolean_pb2, clock_pb2
from gz.msgs10.double_pb2 import Double
from gz.transport13 import Node as GzNode
from attack.profiled_motion_model import profiledMotionModels

class GzTransportController:
    def __init__(self, world_name="default"):
        self.world_name = world_name
        self.node = transport.Node()
        self.service_name = f"/world/{world_name}/control"
        print("Gazebo Transport Controller initialized.")

    def control_simulation(self, action):
        """Pause the simulation using gz-transport"""
        # Create WorldControl message
        req = world_control_pb2.WorldControl()
        if action == "pause":
            req.pause = True
        elif action == "resume":
            req.pause = False
        else:
            print("Invalid action. Use 'pause' or 'resume'.")
            return False

        # Call service with correct signature
        result, response = self.node.request(
            self.service_name,  # service name
            req,  # request message
            world_control_pb2.WorldControl,  # request type
            boolean_pb2.Boolean,  # response type
            3000,  # timeout in ms
        )
        return result, response

    def pause_simulation(self):
        """Pause the simulation using gz-transport"""
        result, response = self.control_simulation("pause")
        if not result:
            return False
        return response.data

    def resume_simulation(self):
        """Resume the simulation using gz-transport"""
        result, response = self.control_simulation("resume")
        if not result:
            return False
        return response.data


class GzClockSubscriber:
    def __init__(self):
        self.node = transport.Node()
        self.latest_sim_time = None
        self.latest_real_time = None
        self.latest_system_time = None

        # Subscribe to clock topic
        self.node.subscribe(clock_pb2.Clock, "/clock", self.clock_callback)
        print("Subscribed to /clock topic")

    def clock_callback(self, msg):
        """Callback for clock messages"""
        # Extract simulation time
        self.latest_sim_time = msg.sim.sec + msg.sim.nsec / 1e9

        # Extract real time
        self.latest_real_time = msg.real.sec + msg.real.nsec / 1e9

        # Extract system time
        self.latest_system_time = msg.system.sec + msg.system.nsec / 1e9

        # Optional: print for debugging
        # print(f"Sim time: {self.latest_sim_time:.3f}s, "
        #       f"Real time: {self.latest_real_time:.3f}s, "
        #       f"System time: {self.latest_system_time:.3f}s")

    def get_sim_time(self):
        """Get latest simulation time"""
        return self.latest_sim_time

    def get_real_time(self):
        """Get latest real time"""
        return self.latest_real_time

    def get_system_time(self):
        """Get latest system time"""
        return self.latest_system_time

    def wait_for_time(self, timeout=5.0):
        """Wait until we receive time data"""
        start = time.time()
        while self.latest_sim_time is None and (time.time() - start) < timeout:
            time.sleep(0.01)
        return self.latest_sim_time is not None


class GzGimbalController:
    def __init__(self, gz_clock, init_pitch=-1.0, **kwargs):
        # Gazebo Transport Node
        self.gz_node = GzNode()
        self.pitch_pub = self.gz_node.advertise(
            "/model/x500_gimbal_0/command/gimbal_pitch", Double
        )
        self.yaw_pub = self.gz_node.advertise(
            "/model/x500_gimbal_0/command/gimbal_yaw", Double
        )
        self.roll_pub = self.gz_node.advertise(
            "/model/x500_gimbal_0/command/gimbal_roll", Double
        )

        # self.pitch_limits = (-2.35, 1.22)
        self.pitch_limits = (-1.60, 0.6)
        self.yaw_limits = (-math.pi, math.pi)
        self.roll_limits = (-0.5, 0.5)

        # PID Controllers for gimbal adjustment
        self.pitch_pid = PID(
            0.0001,
            0.000001,
            0.000001,
            setpoint=0,
            output_limits=self.pitch_limits,
            starting_output=-0.0,
        )
        self.yaw_pid = PID(
            0.0001,
            0.000001,
            0.000001,
            setpoint=0,
            output_limits=self.yaw_limits,
            starting_output=0.0,
        )
        self.roll_pid = PID(
            0.0001,
            0.00001,
            0.00001,
            setpoint=0,
            output_limits=self.roll_limits,
            starting_output=0.0,
        )

        # Allow time for publishers to initialize
        time.sleep(5.0)

        # After publisher established, initialize gimbal angles
        init = Double()
        init.data = init_pitch
        self.pitch_pub.publish(init)
        self.curr_pitch = init.data

        init.data = self.yaw_pid(0)
        self.yaw_pub.publish(init)
        self.curr_yaw = init.data

        init.data = self.roll_pid(0)
        self.roll_pub.publish(init)
        self.curr_roll = init.data
        print("Gimbal controller initialized with default angles.")
        self.init_set = (self.curr_roll, self.curr_pitch, self.curr_yaw)
        
        self.gz_clock = gz_clock
        self.profiled_motion_models = profiledMotionModels()
        # Create a timer for periodic control (0.1 seconds = 10 Hz)
        self.last_control_sim_time = None
        self.last_control_real_time = None
        self.last_control_system_time = None
        self.period_interval = 1/50  # seconds
        self.fps = kwargs.get("fps", 30)
        self.attack_interval_method = kwargs.get("attack_interval_method", "simdt") # "simdt" or "1overfps"
        # self.control_timer = self.gz_node.create_timer(self.period_interval, self.periodic_control_callback)
        self.real_time_factor = kwargs.get("real_time_factor", 0.5)  # Real-time factor for simulation speed adjustments
        self.attack_active = False

        self.sim_dt_history = [1/self.fps]

    def set_wg(self, optimal_offset):
        self.profiled_motion_models.select_resonant_freq(optimal_offset)
        self.profiled_motion_models.select_amplitude(optimal_offset)

    def start_motion(self):
        self.attack_active = True

    def stop_motion(self):
        self.attack_active = False
    
    def reinitialize_gimbal(self):
        """Reinitialize gimbal to default angles"""
        init = Double()
        init.data = self.init_set[1]  # Default pitch angle
        self.pitch_pub.publish(init)
        self.curr_pitch = init.data

        init.data = self.init_set[2]  # Default yaw angle
        self.yaw_pub.publish(init)
        self.curr_yaw = init.data

        init.data = self.init_set[0]  # Default roll angle
        self.roll_pub.publish(init)
        self.curr_roll = init.data

        self.attack_active = False

    def control_callback(self, optimal_velocity=None):
        curr_sim_time = self.gz_clock.get_sim_time() # gazebo sim time after real time factor applied
        curr_real_time = self.gz_clock.get_real_time()
        curr_system_time = self.gz_clock.get_system_time()
        if (
            self.last_control_sim_time is None
            or self.last_control_real_time is None
            or self.last_control_system_time is None
        ):
            self.last_control_sim_time = curr_sim_time
            self.last_control_real_time = curr_real_time
            self.last_control_system_time = curr_system_time
            return [None, None, None], [None, None, None], [self.curr_roll, self.curr_pitch, self.curr_yaw]
        if not self.attack_active:
            return [None, None, None], [None, None, None], [self.curr_roll, self.curr_pitch, self.curr_yaw]

        sim_dt = curr_sim_time - self.last_control_sim_time
        real_dt = curr_real_time - self.last_control_real_time
        system_dt = curr_system_time - self.last_control_system_time
        self.last_control_sim_time = curr_sim_time
        self.last_control_real_time = curr_real_time
        self.last_control_system_time = curr_system_time
        # print(f"Sim dt: {sim_dt:.3f}s, Real dt: {real_dt:.3f}s, System dt: {system_dt:.3f}s")
        self.sim_dt_history.append(sim_dt)

        if optimal_velocity is None:
            angular_velocity = self.profiled_motion_models.get_angular_velocity(curr_sim_time)
            
            attack_interval = 0
            if self.attack_interval_method == "simdt":
                attack_interval = sim_dt
            elif self.attack_interval_method == "1overfps":
                attack_interval = 1/self.fps
            else:
                raise ValueError("Invalid attack_interval_method. Use 'simdt' or '1overfps'.")
            attack_interval = min(attack_interval, 1/10) # fix unrealistic simulation spikes

            position_delta = self.profiled_motion_models.get_position_delta(
                curr_sim_time, dt=attack_interval
            )
            print(f"Gimbal position delta (rad): Roll: {position_delta[0]:.4f}, Pitch: {position_delta[1]:.4f}, Yaw: {position_delta[2]:.4f}")
        else:
            angular_velocity = optimal_velocity
            position_delta = [
                angular_velocity[0] * (1/self.fps),
                angular_velocity[1] * (1/self.fps),
                angular_velocity[2] * (1/self.fps),
            ]
            print(f"Gimbal position delta (rad): Roll: {position_delta[0]:.4f}, Pitch: {position_delta[1]:.4f}, Yaw: {position_delta[2]:.4f}")

        pitch_command = Double()
        yaw_command = Double()
        roll_command = Double()

        pitch_command.data = self.curr_pitch + position_delta[1]
        yaw_command.data = self.curr_yaw + position_delta[2]
        roll_command.data = self.curr_roll + position_delta[0]

        pitch_command.data = max(
            min(pitch_command.data, self.pitch_limits[1]), self.pitch_limits[0]
        )
        yaw_command.data = max(
            min(yaw_command.data, self.yaw_limits[1]), self.yaw_limits[0]
        )
        roll_command.data = max(
            min(roll_command.data, self.roll_limits[1]), self.roll_limits[0]
        )

        self.curr_pitch = pitch_command.data
        self.curr_yaw = yaw_command.data
        self.curr_roll = roll_command.data
        curr_gimbal_setpoint = (self.curr_roll, self.curr_pitch, self.curr_yaw)
        print(f"Current gimbal setpoint (rad): Roll: {self.curr_roll:.4f}, Pitch: {self.curr_pitch:.4f}, Yaw: {self.curr_yaw:.4f}")

        self.pitch_pub.publish(pitch_command)
        self.yaw_pub.publish(yaw_command)
        self.roll_pub.publish(roll_command)
        
        return angular_velocity, position_delta, curr_gimbal_setpoint

    def hit_physical_constraints(self):
        """Check if gimbal angles hit physical constraints"""
        pitch, yaw, roll = False, False, False
        if (
            self.curr_pitch == self.pitch_limits[0]
            or self.curr_pitch == self.pitch_limits[1]
        ):
            print("Gimbal pitch hit physical constraints!")
            pitch = True
        if self.curr_yaw == self.yaw_limits[0] or self.curr_yaw == self.yaw_limits[1]:
            print("Gimbal yaw hit physical constraints!")
            yaw = True
        if (
            self.curr_roll == self.roll_limits[0]
            or self.curr_roll == self.roll_limits[1]
        ):
            print("Gimbal roll hit physical constraints!")
            roll = True
        # return True if any two angles hit constraints
        return (pitch and yaw) or (yaw and roll) or (roll and pitch)


if __name__ == "__main__":
    # # Usage
    # clock_sub = GzClockSubscriber()

    # # Wait for first message
    # if clock_sub.wait_for_time():
    #     print(f"Current simulation time: {clock_sub.get_sim_time():.3f}s")
    # else:
    #     print("Timeout waiting for clock data")

    # # Keep running to receive updates
    # try:
    #     while True:
    #         time.sleep(0.1)  # Let callbacks run
    #         if clock_sub.get_sim_time() is not None:
    #             print(f"Sim: {clock_sub.get_sim_time():.3f}s")
    # except KeyboardInterrupt:
    #     print("Stopping clock subscriber")

    transport_ctrl = GzTransportController()
    result = transport_ctrl.pause_simulation()
    if result:
        print("Simulation paused")
    time.sleep(10)
    result = transport_ctrl.resume_simulation()
    if result:
        print("Simulation resumed")
