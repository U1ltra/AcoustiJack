import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lombscargle
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse

plt.rcParams.update({
    'font.size': 14,
    #'legend.fontsize': 14
})

def process(folder_path):
    amp_stats = {}
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        df = pd.read_csv(file)

        print(f'Filename: {os.path.basename(file)} \n')

        # Convert angular velocities to degrees per second
        gyro = df[['gx', 'gy', 'gz']].values / 131.2

        # Apply rotation matrix to gyro data
        # R = np.load('imu_rotation_matrix.npy')
        # gyro = np.dot(gyro, R.T)

        # Remove DC bias (subtract mean from each axis)
        # gyro -= np.mean(gyro, axis=0)

        # Convert nanosecond timestamps to datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        dt_mean = np.mean(df['timestamp'].diff().dt.total_seconds().values[1:])
            
        # Apply low-pass filter to gyro data
        # b, a = butter(N=4, Wn=(2.0 / (0.5 / dt_mean)), btype='low')
        # gyro = np.array([filtfilt(b, a, gyro[:, i]) for i in range(3)]).T

        # gyro[1:] = np.cumsum(gyro[1:] * dt_mean, axis=0)
        df[['gx', 'gy', 'gz']] = gyro

        # Compute magnitude of filtered gyro data
        mag = np.linalg.norm(df[['gx', 'gy', 'gz']], axis=1)

        # Compute rolling RMS on raw magnitude
        window_size = int(0.01 / dt_mean)  # 10 ms window
        rms = pd.Series(mag).rolling(window_size, center=True).mean().fillna(0)

        # Threshold to detect motion
        osc_idx = np.where(rms > np.percentile(rms, 60))[0]

        # Estimate number of motion samples
        n_samp = int(len(osc_idx))

        print(f'Window length: {len(osc_idx) * dt_mean}')

        # Collect n samples around the center of oscillation
        if len(osc_idx) > 0:
            center_idx = osc_idx[len(osc_idx) // 2]
            df = df.iloc[(center_idx - n_samp // 2):(center_idx + n_samp // 2)].reset_index(drop=True)
        else:
            print('Error in oscillation detection for file ', os.path.basename(file))

        # Create a single figure with two vertically stacked subplots
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 4), sharex=False, 
                                    gridspec_kw={'hspace': 0.1})

        # Raw data plot
        t_plot = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds().values
        ax1.plot(t_plot, df['gx'], label='Roll')
        ax1.plot(t_plot, df['gy'], label='Pitch')
        ax1.plot(t_plot, df['gz'], label='Yaw')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), ncol=3, fontsize=18)
        ax1.set_xticklabels([])  # Remove x-axis labels from top plot
        ax1.tick_params(axis='both', which='major', labelsize=16)

        # Prepare for reconstructed signals plot
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        t = t_plot  # Use the same time values
        freqs = np.linspace(0.1, 0.5 / dt_mean, 5000)
        angular_freqs = 2 * np.pi * freqs

        # Find min/max for y axis across raw and reconstructed signals
        y_min = np.min(df[['gx', 'gy', 'gz']].values)
        y_max = np.max(df[['gx', 'gy', 'gz']].values)
        recon_min = []
        recon_max = []
        total_amp = 0
        amps = {}  # Store amplitudes for this file

        for axis in ['gx', 'gy', 'gz']:
            y = df[axis].values
            y = y - np.mean(y) 

            # Lomb-Scargle power spectrum
            power = lombscargle(t, y, angular_freqs, precenter=False)

            peak_idx = np.argmax(power)
            freq = freqs[peak_idx]
            omega = angular_freqs[peak_idx]

            # Fit phase
            cos_term = np.cos(omega * t)
            sin_term = np.sin(omega * t)
            A_mat = np.vstack([cos_term, sin_term]).T
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
            a, b = coeffs
            amp = np.sqrt(a**2 + b**2)
            phase = -np.arctan2(b, a)

            # Reconstruct cosine signal
            recon = amp * np.cos(omega * t + phase) * 0.1
            ax2.plot(t, recon, label="_nolegend_")
            recon_min.append(np.min(recon))
            recon_max.append(np.max(recon))

            amps[axis] = amp
            avg_err = np.mean(np.abs(recon - y))
            print(f"Position scalar: {amp/omega}")
            print(f"{axis} | Frequency: {freq:.4f} Hz, Amplitude: {amp:.4f}, Phase: {phase:.4f} rad, AvgErr: {avg_err:.4f}")
            total_amp += amp

        # Set y-axis limits to be the same for both plots, with vertical padding
        global_min = min(y_min, min(recon_min))
        global_max = max(y_max, max(recon_max))
        y_range = global_max - global_min
        pad = 0.05 * y_range if y_range > 0 else 1.0
        ax1.set_ylim(global_min - pad, global_max + pad)
        ax2.set_ylim(global_min - pad, global_max + pad)

        # Add single y-label for both subplots
        # fig.text(0.025, 0.5, r'Angular Velocity (°/s)', va='center', rotation='vertical', fontsize=18)

        # Store the amplitude and frequency data
        amp_stats[os.path.basename(file)] = (amps, freq)

        # Save the combined figure to a single-page PDF
        pdf_prefix = os.path.splitext(os.path.basename(file))[0]
        pdf_path = f"{pdf_prefix}_stacked.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print('-' * 50)
        
        plt.show()
        
    print("\nAverage max velocity (amplitude) and stddev per axis across files:")
    axes = ['gx', 'gy', 'gz']
    for axis in axes:
        vals = [amp_stats[f][0][axis] for f in amp_stats]
        print(f"{axis}: mean = {np.mean(vals):.4f}, std = {np.std(vals):.4f}")

    print(f"Average frequency: {np.mean([amp_stats[f][1] for f in amp_stats]):.4f} Hz")

    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process IMU data for spectrum analysis.')
    parser.add_argument('folder', type=str, help='Path to the folder containing CSV files')
    args = parser.parse_args()
    process(args.folder)
