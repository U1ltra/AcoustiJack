#region Imports
from AD9833_spi import AD9833
from BMI160_i2c import BMI160, definitions
from RIGOL_scpi import DG1022

import board
import busio
import adafruit_ds3502

import pyaudio
import wave

from gpiozero import Servo

import os
import csv
import time

from datetime import datetime
from enum import Enum

import threading
import argparse

import socket
import struct
#endregion

class InjMethod(Enum):
    RIGOL = 'rigol'
    MINI = 'AD9833'
    NA = 'none'

class MotMethod(Enum):
    IMU = 'bmi160'
    PX4 = 'px4'
    VIZ = 'vision'
    NA = 'none'
        
class Controller:
    def __init__(self, use_mic: bool, method: InjMethod, motion: MotMethod):
        #region Internal variable declarations
        self._starting_gun = threading.Event()
        self._logging = False
        self._header = ["time", "gx", "gy", "gz", "ax", "ay", "az", "temp"]
        self._audio_frames = []
        self._use_mic = use_mic
        self._method = method
        self._motion = motion
        self._dir = 0
        #endregion

        #region Initialize motion logging
        if self._motion == MotMethod.IMU:
            try: self.imu = BMI160(); print('[Controller] BMI160 Initialized')
            except Exception as e: print(f'[Controller] BMI160 Initialization Failed: {e}'); exit()
        elif self._motion == MotMethod.PX4:
            print('PX4')
        elif self._motion == MotMethod.VIZ:
            try: (self._sock := socket.socket(socket.AF_INET, socket.SOCK_DGRAM)).bind(("", 5555)); print('[Controller] Vision initialized')
            except Exception as e: print(f'[Controller] Vision initialization failed: {e}'); exit()
        #endregion

        #region Initialize function gen pipeline
        if self._method == InjMethod.MINI:
            try: self.gen = AD9833(); print('[Controller] AD9833 Initialized')
            except Exception as e: print(f'[Controller] AD9833 Initialization Failed: {e}'); exit()
            try: self.pot = adafruit_ds3502.DS3502(busio.I2C(board.SCL, board.SDA)); print('[Controller] DS3502 Initialized')
            except Exception as e: print(f'[Controller] DS3502 Initialization Failed: {e}'); exit()
        elif self._method == InjMethod.RIGOL:
            try: self.inst = DG1022(); print('[Controller] RIGOL DG1022 Initialized')
            except Exception as e: print(f'[Controller] RIGOL DG1022 Initialization Failed: {e}'); exit()
        #endregion

        #region Initialize audio recording
        if self._use_mic:
            try:
                self._audio = pyaudio.PyAudio()
                self._stream = self._audio.open(format=pyaudio.paInt16,
                                               channels=1,
                                               rate=44100,
                                               input=True,
                                               frames_per_buffer=1024,
                                               start=False)
                print('[Controller] Audio Recording Initialized')
            except Exception as e: print(f"[Controller] Audio Recording Initialization Failed: {e}"); exit()
        #endregion

        #region Create log directory
        if self._use_mic or self._motion in [MotMethod.IMU, MotMethod.PX4]:
            self.dir = os.path.join("logs", datetime.now().strftime("%Y%m%d"))
            print(f"[Controller] Log directory created at {self.dir}")
            os.makedirs(self.dir, exist_ok=True)
        #endregion

    def _mic_worker(self, filename):
        self._stream.start_stream()
        self._starting_gun.wait()
        
        while self._logging:
            self._audio_frames.append(self._stream.read(1024, exception_on_overflow=False))
            
        self._stream.stop_stream()

        with wave.open(audio_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self._audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self._audio_frames))
            print(f"[Controller] Audio saved to {audio_filename}")
            
        self._stream.close()
        self._audio.terminate()
        
    def _imu_worker(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._header)

            # calibrate
            self._imu_offset = list(self.imu.getMotion6())
            self._imu_offset[:3] = [value / 131.2 for value in self._imu_offset[:3]]
            self._imu_offset[3:] = [value / 16384.0 for value in self._imu_offset[3:]]
            
            print(f"[Controller] IMU calibration complete, offset: {self._imu_offset}")
            
            self._starting_gun.wait()
            while self._logging:
                data = list(self.imu.getMotion6())
                data[:3] = [value / 131.2 for value in data[:3]]
                data[3:] = [value / 16384.0 for value in data[3:]]
                
                if self._imu_offset is not None: data = [data[i] - self._imu_offset[i] for i in range(6)]
                data.append(self.imu.getTemperature())

                self._dir = data[2]

                writer.writerow([time.perf_counter_ns(), *data])
        
        print(f"[Controller] Motion data saved to {motion_filename}")
        self.imu.close()

    def _px4_worker(self, filename):
        print('px4')

    def _viz_worker(self):
        self._starting_gun.wait()
        while self._logging:
            data, addr = self._sock.recvfrom(1024)
            orient_values = struct.unpack('3f', data)
            self._dir = orient_values[2]
        
    def _flip_worker(self):
        self._flipper = lambda theta: bool(theta > 0)
        self._flip_int = 0.25
        self._last_flip = -self._flip_int
        
        if self._method == InjMethod.RIGOL: fg = self.inst
        elif self._method == InjMethod.MINI: fg = self.gen
        else: fg = type('', (), {'flip': lambda self: None})()

        self._starting_gun.wait()
        while self._logging:
            time_ok = (time.perf_counter() - self._last_flip) > self._flip_int
            if self._flipper(self._dir) and time_ok: fg.flip(); self._last_flip = time.perf_counter();
        
    #region Description
    # Method is a value in enum Method. 
    # Freq is a float with up to one decimal of precision. 
    # Amp is an integer 0-127 for DS3502, 0-20 for RIGOL
    # Duration is a integer denoting seconds.
    # Act_flipper is a boolean denoting whether or not the phase control is active.
    #endregion
    def inject(self, freq: float, amp: int, duration: int, act_flipper: bool):
        #region Generate filename
        timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        paramstring = f"{self._method.value}-f{freq:.1f}-a{amp}-{duration}s-dt{timestr}".replace(".", "_")
        #endregion

        #region Start logging data
        if self._motion == MotMethod.IMU:
            motthread = threading.Thread(target=self._imu_worker, args=(os.path.join(self.dir, f"{paramstring}.csv"),))
        elif self._motion == MotMethod.PX4:
            motthread = threading.Thread(target=self._px4_worker, args=(os.path.join(self.dir, f"{paramstring}.csv"),))
        elif self._motion == MotMethod.VIZ:
            motthread = threading.Thread(target=self._viz_worker)
        else: motthread = threading.Thread(target=lambda: None)
        motthread.start()

        if act_flipper: flipthread = threading.Thread(target=self._flip_worker, args=(self._method))
        else: flipthread = threading.Thread(target=lambda: None)
        flipthread.start()

        if self._use_mic: micthread = threading.Thread(target=self._mic_worker, args=(os.path.join(self.dir, f"{paramstring}.wav"),))
        else: micthread = threading.Thread(target=lambda: None)
        micthread.start()
        
        self._logging = True
        self._starting_gun.set()
        
        time.sleep(1)
        #endregion

        #region Execute injection
        if self._method == InjMethod.MINI:
            self.pot.wiper = amp
            self.gen.start(freq)
            print("[Controller] AD9833 injection started")
            time.sleep(duration)
            self.gen.stop()
        elif self._method == InjMethod.RIGOL:
            self.inst.start(freq, amp)
            print("[Controller] DG1022 injection started")
            time.sleep(duration)
            self.inst.stop()
            self.inst.close()
        else: time.sleep(duration)
        #endregion

        #region Stop logging data
        time.sleep(1)
        self._logging = False
        
        motthread.join()
        flipthread.join()
        micthread.join()
        #endregion

if __name__ == "__main__":
    #region Parse arguments
    parser = argparse.ArgumentParser(
                    prog='U-M RobustNet Lab Acoustic Injection System Controller',
                    description='Manages audio and motion recording, as well as piezo execution of acoustic injection attacks.')

    parser.add_argument('--method', type=str, required=True, help='Injection generation method')
    parser.add_argument('--freq', type=float, required=True, help='Frequency of the injection in Hz, up to one decimal of precision for AD9833')
    parser.add_argument('--amp', type=int, required=True, help='Amplitude of the injection, 0-127 for AD9833')
    parser.add_argument('--duration', type=int, required=True, help='Duration of the injection, in seconds')
    parser.add_argument('--no-mic', action='store_true', help='Disable audio recording and skip writing .wav file')
    parser.add_argument('--motion', type=str, required=True, help='Motion logging method')
    parser.add_argument('--flipper', action='store_true', help='Activate flipper')

    args = parser.parse_args()
    #endregion

    #region Execute
    controller = Controller((not args.no_mic), InjMethod[args.method], MotMethod[args.motion])
    controller.inject(args.freq, args.amp, args.duration, args.flipper)
    #endregion














