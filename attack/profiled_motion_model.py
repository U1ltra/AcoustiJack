import os
import math
import numpy as np


class MotionWaveGenerator:
    """
    A class to generate various wave patterns for adversarial motion injection.
    """

    def __init__(
        self,
        resonant_freq=7744.0,  # Hz
        max_amplitude=2.0,  # rad/s
        aliased_frequency=1.0,  # Hz
        initial_phase=0.0,  # radians
        wave_type="sine",  # 'sine'
        std=math.radians(
            8
        ),  # standard deviation of noise added to angular velocity (rad/s)
    ):
        """
        Initialize the wave generator.

        Args:
            frequency (float): Frequency of the wave in Hz
            max_amplitude (float): Maximum amplitude of the wave
            wave_type (str): Type of wave ('sine')
        """
        self.resonant_freq = resonant_freq
        self.frequency = aliased_frequency
        self.max_amplitude = max_amplitude
        self.initial_max_amplitude = max_amplitude
        self.initial_phase = initial_phase
        self.wave_type = wave_type
        self.is_active = True
        self.noise_std = std
        self.interval = (
            1 / 50
        )  # Default to 50 Hz update rate for the gimbal motion attack simulation

    def get_angular_velocity(self, t):
        """
        Calculate the angular velocity at time t.

        Args:
            t (float): Time in seconds. If None, uses current time.

        Returns:
            float: Angular velocity in rad/s
        """
        if not self.is_active:
            return 0.0

        if self.wave_type == "sine":
            return self.max_amplitude * math.sin(
                2 * math.pi * self.frequency * t + self.initial_phase
            )
        else:
            raise ValueError(f"Unknown wave type: {self.wave_type}")

    def get_position_delta(self, t, dt=None):
        """
        Calculate the position delta by integrating angular velocity over time dt.

        Args:
            dt (float): Time step in seconds
            t (float): Current time. If None, uses current time.

        Returns:
            float: Position delta in radians
        """
        if not self.is_active or dt is None or dt <= 0:
            return 0.0

        angular_velocity = self.get_angular_velocity(t)
        angular_velocity += np.random.normal(0, self.noise_std)
        if dt is not None:
            return angular_velocity * dt
        else:
            return angular_velocity * self.interval

    def get_half_period_delta(self):
        """
        Calculate the position delta over half a period of the wave.

        Returns:
            float: Position delta in radians
        """
        if not self.is_active:
            return 0.0

        half_period = 1 / (2 * self.frequency)
        sample_num = max(1, int(half_period / self.interval))
        half_period = sample_num * self.interval
        offset = 0.0
        for t in np.linspace(0, half_period, sample_num):
            angular_velocity = self.get_angular_velocity(t)
            angular_velocity += np.random.normal(0, self.noise_std)
            offset += angular_velocity * self.interval

        return offset
    
    def in_first_half_period(self, t):
        period = 1 / self.frequency
        half_period = period / 2
        t_mod = t % period
        return t_mod <= half_period


class profiledMotionModels:
    def __init__(self):
        self.resonant_freqs = [7744.0, 7746.0, 23231.0, 30000.0]  # Hz
        self.aliased_frequencies = [1.0273, 4.9739, 2.0332, 4.0]  # Hz
        self.velocity_amplitudes = [  # roll, pitch, yaw for each resonant frequency
            [math.radians(3.8690), math.radians(17.1677), math.radians(87.2845)],
            [math.radians(4.5054), math.radians(12.1534), math.radians(92.8463)],
            [math.radians(2.0032), math.radians(15.6742), math.radians(134.5029)],
            # [math.radians(2.0032), math.radians(134.5029), math.radians(15.6742)],
        ]
        self.resonant_freqs = self.resonant_freqs[: len(self.velocity_amplitudes)]
        self.aliased_frequencies = self.aliased_frequencies[: len(self.velocity_amplitudes)]
        self.unit_directions = []
        self.offset_norms = []
        self.wave_generators = []
        for i in range(len(self.resonant_freqs)):
            roll_wg = MotionWaveGenerator(
                resonant_freq=self.resonant_freqs[i],
                max_amplitude=self.velocity_amplitudes[i][0],
                aliased_frequency=self.aliased_frequencies[i],
                initial_phase=0.0,
                wave_type="sine",
            )
            pitch_wg = MotionWaveGenerator(
                resonant_freq=self.resonant_freqs[i],
                max_amplitude=self.velocity_amplitudes[i][1],
                aliased_frequency=self.aliased_frequencies[i],
                initial_phase=0.0,
                wave_type="sine",
            )
            yaw_wg = MotionWaveGenerator(
                resonant_freq=self.resonant_freqs[i],
                max_amplitude=self.velocity_amplitudes[i][2],
                aliased_frequency=self.aliased_frequencies[i],
                initial_phase=0.0,
                wave_type="sine",
            )
            self.wave_generators.append([roll_wg, pitch_wg, yaw_wg])

            direction = np.array(
                [
                    roll_wg.get_half_period_delta(),
                    pitch_wg.get_half_period_delta(),
                    yaw_wg.get_half_period_delta(),
                ]
            )
            norm = np.linalg.norm(direction)
            self.offset_norms.append(norm)
            direction = direction / norm
            self.unit_directions.append(direction)

        self.unit_directions = np.array(self.unit_directions)

        self.curr_wg = None
        self.curr_wg_idx = -1
        self.curr_sign = 1

        self.lost_angles = []
        self.lost_norms = []

    def select_resonant_freq(self, opt_attack_offset):
        unit_attack_offset = opt_attack_offset / np.linalg.norm(opt_attack_offset)
        cos_sims = np.dot(self.unit_directions, unit_attack_offset)
        print("Cosine similarities with resonant freq directions: ", cos_sims)
        best_idx = np.argmax(np.abs(cos_sims)) # select the most aligned direction
        self.curr_wg_idx = best_idx
        self.curr_wg = self.wave_generators[best_idx]
        self.curr_sign = np.sign(cos_sims[best_idx])
        print(f"Selected resonant frequency: {self.resonant_freqs[best_idx]} Hz, aliased frequency: {self.aliased_frequencies[best_idx]} Hz, direction: {self.unit_directions[best_idx]}, sign: {self.curr_sign}")

        # calculate the angle between the unit attack offset and the selected direction
        angle = math.acos(cos_sims[best_idx]) * 180 / math.pi
        self.lost_angles.append(angle)

        return self.curr_wg
    
    def select_amplitude(self, opt_attack_offset):
        if self.curr_wg is None or self.curr_wg_idx < 0:
            raise ValueError("No resonant frequency selected yet.")
        # project the optimal attack offset to the selected direction
        direction = self.unit_directions[self.curr_wg_idx]
        proj_length = np.dot(opt_attack_offset, direction)
        scale = np.abs(proj_length / self.offset_norms[self.curr_wg_idx])
        
        offset_norm = np.linalg.norm(opt_attack_offset)
        proj_to_offset_ratio = np.abs(proj_length / offset_norm)
        self.lost_norms.append(proj_to_offset_ratio)

        for wg in self.curr_wg:
            # scaling the max amplitude directly results same proportion scale in the integrated offset
            wg.initial_max_amplitude *= scale
            
    def get_angular_velocity(self, t):
        if self.curr_wg is None:
            return np.array([0.0, 0.0, 0.0])
        omega = np.array(
            [
                self.curr_wg[0].get_angular_velocity(t),
                self.curr_wg[1].get_angular_velocity(t),
                self.curr_wg[2].get_angular_velocity(t),
            ]
        )
        omega *= self.curr_sign  # simulated phase shifting
        if not self.curr_wg[2].in_first_half_period(t):
            omega *= -1  # invert direction in the second half period

        return omega
            
    def get_position_delta(self, t, dt):
        if self.curr_wg is None:
            return np.array([0.0, 0.0, 0.0])
        delta = np.array(
            [
                self.curr_wg[0].get_position_delta(t, dt),
                self.curr_wg[1].get_position_delta(t, dt),
                self.curr_wg[2].get_position_delta(t, dt),
            ]
        )

        delta *= self.curr_sign  # simulated phase shifting
        if not self.curr_wg[2].in_first_half_period(t):
            delta *= -1  # invert direction in the second half period

        return delta

    def save_traces(self, save_dir):
        np.save(os.path.join(save_dir, "lost_angles.npy"), np.array(self.lost_angles))
        np.save(os.path.join(save_dir, "lost_norms.npy"), np.array(self.lost_norms))