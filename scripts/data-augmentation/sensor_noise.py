#!/usr/bin/env python3
"""
Sensor Noise Module for Rocket Flight Data
Implements realistic IMU sensor noise models including:
- Accelerometer noise (white noise, bias, temperature drift)
- Gyroscope noise (white noise, bias drift, scale factor errors)
- Barometer/altimeter noise (pressure noise, quantization)
- GPS noise (position and velocity errors)
- Sensor dropout and glitches

Based on typical MEMS IMU characteristics (e.g., MPU6050, BNO055, BMI088)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


# ============================================================
# SENSOR NOISE CONFIGURATIONS
# ============================================================

@dataclass
class AccelerometerNoise:
    """
    Accelerometer noise parameters (typical MEMS IMU)
    Based on datasheets: MPU6050, ADXL345, LSM6DS3
    """
    # White noise (noise density in µg/√Hz converted to m/s²/√Hz)
    noise_density: float = 400e-6 * 9.81  # 400 µg/√Hz → m/s²/√Hz
    
    # Bias instability (m/s²)
    bias_instability: float = 0.02  # ~2 mg
    
    # Bias random walk (m/s²/√s)
    bias_random_walk: float = 0.001
    
    # Initial bias offset (m/s²)
    initial_bias_std: float = 0.05  # ~5 mg
    
    # Temperature drift (m/s²/°C)
    temp_drift: float = 0.01
    
    # Scale factor error (%)
    scale_factor_error: float = 0.02  # 2%
    
    # Saturation limits (m/s²)
    saturation_limit: float = 160.0  # ±16g typical
    
    # Quantization (bits)
    resolution_bits: int = 16


@dataclass
class GyroscopeNoise:
    """
    Gyroscope noise parameters (typical MEMS IMU)
    Based on datasheets: MPU6050, L3GD20, BMI088
    """
    # White noise (deg/s/√Hz converted to rad/s/√Hz)
    noise_density: float = 0.01 * np.pi / 180  # 0.01 deg/s/√Hz
    
    # Bias instability (rad/s)
    bias_instability: float = 0.01 * np.pi / 180  # 0.01 deg/s
    
    # Bias random walk (rad/s/√s)
    bias_random_walk: float = 0.0001 * np.pi / 180
    
    # Initial bias offset (rad/s)
    initial_bias_std: float = 0.1 * np.pi / 180  # 0.1 deg/s
    
    # Temperature drift (rad/s/°C)
    temp_drift: float = 0.02 * np.pi / 180
    
    # Scale factor error (%)
    scale_factor_error: float = 0.02  # 2%
    
    # Saturation limits (rad/s)
    saturation_limit: float = 2000 * np.pi / 180  # ±2000 deg/s
    
    # Resolution (bits)
    resolution_bits: int = 16


@dataclass
class BarometerNoise:
    """
    Barometer/altimeter noise parameters
    Based on: MS5611, BMP280, BMP388
    """
    # Pressure noise (Pa)
    pressure_noise: float = 2.0  # ~0.17 m altitude noise
    
    # Altitude quantization (m)
    altitude_resolution: float = 0.01  # 1 cm
    
    # Bias drift (Pa/s)
    bias_drift_rate: float = 0.1
    
    # Temperature sensitivity (Pa/°C)
    temp_sensitivity: float = 0.5


@dataclass
class GPSNoise:
    """
    GPS receiver noise parameters
    Based on: u-blox M8, NEO-7M typical performance
    """
    # Horizontal position accuracy (m, CEP)
    horizontal_accuracy: float = 2.5  # 2.5m CEP typical
    
    # Vertical position accuracy (m)
    vertical_accuracy: float = 5.0
    
    # Velocity accuracy (m/s)
    velocity_accuracy: float = 0.1
    
    # Update rate (Hz)
    update_rate: float = 5.0  # 5 Hz typical
    
    # Multipath error std (m)
    multipath_std: float = 1.0


@dataclass
class SensorDropoutConfig:
    """Configuration for sensor dropout events"""
    # Dropout probability per second
    dropout_rate: float = 0.01  # 1% chance per second
    
    # Dropout duration range (seconds)
    dropout_duration_min: float = 0.01
    dropout_duration_max: float = 0.5
    
    # Glitch probability per second
    glitch_rate: float = 0.005  # 0.5% per second
    
    # Glitch magnitude (multiple of noise std)
    glitch_magnitude: float = 10.0


# ============================================================
# SENSOR NOISE SIMULATOR
# ============================================================

class SensorNoiseSimulator:
    """
    Simulates realistic sensor noise for rocket flight data.
    Applies multiple noise sources with proper correlations.
    """
    
    def __init__(self,
                 accel_config: AccelerometerNoise = None,
                 gyro_config: GyroscopeNoise = None,
                 baro_config: BarometerNoise = None,
                 gps_config: GPSNoise = None,
                 dropout_config: SensorDropoutConfig = None,
                 sample_rate: float = 100.0,
                 seed: int = None):
        """
        Initialize sensor noise simulator
        
        Args:
            accel_config: Accelerometer noise parameters
            gyro_config: Gyroscope noise parameters
            baro_config: Barometer noise parameters
            gps_config: GPS noise parameters
            dropout_config: Dropout configuration
            sample_rate: IMU sample rate (Hz)
            seed: Random seed for reproducibility
        """
        self.accel_config = accel_config or AccelerometerNoise()
        self.gyro_config = gyro_config or GyroscopeNoise()
        self.baro_config = baro_config or BarometerNoise()
        self.gps_config = gps_config or GPSNoise()
        self.dropout_config = dropout_config or SensorDropoutConfig()
        
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize sensor biases (persistent throughout flight)
        self.accel_bias = np.random.normal(0, self.accel_config.initial_bias_std, 3)
        self.gyro_bias = np.random.normal(0, self.gyro_config.initial_bias_std, 3)
        self.baro_bias = 0.0
        
        # Temperature (simulated)
        self.temperature = 25.0  # °C
    
    def add_accelerometer_noise(self, 
                                accel_true: np.ndarray,
                                time: np.ndarray) -> np.ndarray:
        """
        Add realistic accelerometer noise
        
        Args:
            accel_true: True acceleration (N, 3) in m/s²
            time: Time vector (N,)
            
        Returns:
            Noisy acceleration measurements (N, 3)
        """
        n_samples = len(time)
        accel_noisy = accel_true.copy()
        
        # White noise
        white_noise = np.random.normal(
            0, 
            self.accel_config.noise_density * np.sqrt(self.sample_rate),
            (n_samples, 3)
        )
        
        # Bias drift (random walk)
        bias_drift = np.cumsum(
            np.random.normal(0, self.accel_config.bias_random_walk * np.sqrt(self.dt), (n_samples, 3)),
            axis=0
        )
        
        # Temperature drift (simplified sinusoidal)
        temp_variation = 5 * np.sin(2 * np.pi * time / 60)  # ±5°C over 60s
        temp_drift = self.accel_config.temp_drift * temp_variation[:, np.newaxis]
        
        # Scale factor error
        scale_error = 1.0 + np.random.normal(0, self.accel_config.scale_factor_error)
        
        # Combine all noise sources
        accel_noisy = (accel_noisy * scale_error + 
                      self.accel_bias + 
                      bias_drift + 
                      white_noise + 
                      temp_drift)
        
        # Apply saturation
        accel_noisy = np.clip(accel_noisy, 
                             -self.accel_config.saturation_limit,
                             self.accel_config.saturation_limit)
        
        # Quantization
        if self.accel_config.resolution_bits > 0:
            range_val = 2 * self.accel_config.saturation_limit
            lsb = range_val / (2 ** self.accel_config.resolution_bits)
            accel_noisy = np.round(accel_noisy / lsb) * lsb
        
        return accel_noisy
    
    def add_gyroscope_noise(self,
                           gyro_true: np.ndarray,
                           time: np.ndarray) -> np.ndarray:
        """
        Add realistic gyroscope noise
        
        Args:
            gyro_true: True angular velocity (N, 3) in rad/s
            time: Time vector (N,)
            
        Returns:
            Noisy gyroscope measurements (N, 3)
        """
        n_samples = len(time)
        gyro_noisy = gyro_true.copy()
        
        # White noise
        white_noise = np.random.normal(
            0,
            self.gyro_config.noise_density * np.sqrt(self.sample_rate),
            (n_samples, 3)
        )
        
        # Bias drift
        bias_drift = np.cumsum(
            np.random.normal(0, self.gyro_config.bias_random_walk * np.sqrt(self.dt), (n_samples, 3)),
            axis=0
        )
        
        # Temperature drift
        temp_variation = 5 * np.sin(2 * np.pi * time / 60)
        temp_drift = self.gyro_config.temp_drift * temp_variation[:, np.newaxis]
        
        # Scale factor error
        scale_error = 1.0 + np.random.normal(0, self.gyro_config.scale_factor_error)
        
        # Combine noise sources
        gyro_noisy = (gyro_noisy * scale_error +
                     self.gyro_bias +
                     bias_drift +
                     white_noise +
                     temp_drift)
        
        # Apply saturation
        gyro_noisy = np.clip(gyro_noisy,
                            -self.gyro_config.saturation_limit,
                            self.gyro_config.saturation_limit)
        
        # Quantization
        if self.gyro_config.resolution_bits > 0:
            range_val = 2 * self.gyro_config.saturation_limit
            lsb = range_val / (2 ** self.gyro_config.resolution_bits)
            gyro_noisy = np.round(gyro_noisy / lsb) * lsb
        
        return gyro_noisy
    
    def add_barometer_noise(self,
                           altitude_true: np.ndarray,
                           time: np.ndarray) -> np.ndarray:
        """
        Add barometer/altimeter noise
        
        Args:
            altitude_true: True altitude (N,) in meters
            time: Time vector (N,)
            
        Returns:
            Noisy altitude measurements (N,)
        """
        n_samples = len(time)
        
        # White noise
        noise = np.random.normal(0, self.baro_config.pressure_noise / 12.0, n_samples)
        
        # Bias drift
        bias_drift = np.cumsum(
            np.random.normal(0, self.baro_config.bias_drift_rate * self.dt, n_samples)
        ) / 12.0  # Convert pressure to altitude
        
        # Temperature effect
        temp_variation = 5 * np.sin(2 * np.pi * time / 60)
        temp_effect = self.baro_config.temp_sensitivity * temp_variation / 12.0
        
        altitude_noisy = altitude_true + noise + bias_drift + temp_effect
        
        # Quantization
        altitude_noisy = np.round(altitude_noisy / self.baro_config.altitude_resolution) * \
                        self.baro_config.altitude_resolution
        
        return altitude_noisy
    
    def add_gps_noise(self,
                     position_true: np.ndarray,
                     velocity_true: np.ndarray,
                     time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add GPS noise with lower update rate
        
        Args:
            position_true: True position (N, 3) in meters [x, y, z]
            velocity_true: True velocity (N, 3) in m/s
            time: Time vector (N,)
            
        Returns:
            (noisy_position, noisy_velocity) both (N, 3)
        """
        n_samples = len(time)
        
        # GPS updates at lower rate - interpolate between updates
        gps_dt = 1.0 / self.gps_config.update_rate
        gps_indices = np.arange(0, n_samples, int(self.sample_rate / self.gps_config.update_rate))
        
        # Position noise (horizontal and vertical different)
        pos_noise = np.zeros((n_samples, 3))
        for idx in gps_indices:
            if idx < n_samples:
                pos_noise[idx, :2] = np.random.normal(0, self.gps_config.horizontal_accuracy, 2)
                pos_noise[idx, 2] = np.random.normal(0, self.gps_config.vertical_accuracy)
        
        # Forward-fill GPS updates (hold last value)
        for i in range(1, n_samples):
            if i not in gps_indices:
                pos_noise[i] = pos_noise[i-1]
        
        # Velocity noise
        vel_noise = np.zeros((n_samples, 3))
        for idx in gps_indices:
            if idx < n_samples:
                vel_noise[idx] = np.random.normal(0, self.gps_config.velocity_accuracy, 3)
        
        # Forward-fill
        for i in range(1, n_samples):
            if i not in gps_indices:
                vel_noise[i] = vel_noise[i-1]
        
        # Multipath (correlated noise)
        multipath = np.random.normal(0, self.gps_config.multipath_std, 3) * \
                   np.sin(2 * np.pi * time / 10)[:, np.newaxis]
        
        position_noisy = position_true + pos_noise + multipath
        velocity_noisy = velocity_true + vel_noise
        
        return position_noisy, velocity_noisy
    
    def add_dropouts_and_glitches(self,
                                 data: np.ndarray,
                                 time: np.ndarray,
                                 sensor_type: str = 'imu') -> Tuple[np.ndarray, np.ndarray]:
        """
        Add sensor dropouts and occasional glitches
        
        Args:
            data: Sensor data (N, M)
            time: Time vector (N,)
            sensor_type: Type of sensor for determining dropout characteristics
            
        Returns:
            (noisy_data, validity_mask) where validity_mask is False during dropouts
        """
        n_samples = len(time)
        noisy_data = data.copy()
        validity_mask = np.ones(n_samples, dtype=bool)
        
        # Simulate dropouts
        dropout_prob = self.dropout_config.dropout_rate * self.dt
        
        i = 0
        while i < n_samples:
            if np.random.random() < dropout_prob:
                # Dropout event
                duration = np.random.uniform(
                    self.dropout_config.dropout_duration_min,
                    self.dropout_config.dropout_duration_max
                )
                dropout_samples = int(duration * self.sample_rate)
                end_idx = min(i + dropout_samples, n_samples)
                
                # Mark as invalid
                validity_mask[i:end_idx] = False
                
                # During dropout, hold last value or set to NaN
                noisy_data[i:end_idx] = np.nan
                
                i = end_idx
            else:
                i += 1
        
        # Simulate glitches
        glitch_prob = self.dropout_config.glitch_rate * self.dt
        for i in range(n_samples):
            if np.random.random() < glitch_prob and validity_mask[i]:
                # Add large spike
                noise_magnitude = self.dropout_config.glitch_magnitude * data.std(axis=0)
                noisy_data[i] += np.random.choice([-1, 1]) * noise_magnitude
        
        return noisy_data, validity_mask
    
    def simulate_full_imu(self,
                         accel_true: np.ndarray,
                         gyro_true: np.ndarray,
                         altitude_true: np.ndarray,
                         time: np.ndarray,
                         apply_dropouts: bool = True) -> Dict[str, np.ndarray]:
        """
        Simulate complete IMU sensor suite with all noise sources
        
        Args:
            accel_true: True acceleration (N, 3)
            gyro_true: True angular velocity (N, 3)
            altitude_true: True altitude (N,)
            time: Time vector (N,)
            apply_dropouts: Whether to simulate dropouts
            
        Returns:
            Dictionary with noisy sensor data and validity masks
        """
        # Add noise to each sensor
        accel_noisy = self.add_accelerometer_noise(accel_true, time)
        gyro_noisy = self.add_gyroscope_noise(gyro_true, time)
        altitude_noisy = self.add_barometer_noise(altitude_true, time)
        
        result = {
            'time': time,
            'accel_x': accel_noisy[:, 0],
            'accel_y': accel_noisy[:, 1],
            'accel_z': accel_noisy[:, 2],
            'gyro_x': gyro_noisy[:, 0],
            'gyro_y': gyro_noisy[:, 1],
            'gyro_z': gyro_noisy[:, 2],
            'altitude': altitude_noisy,
        }
        
        # Apply dropouts if requested
        if apply_dropouts:
            # IMU dropouts (affect accel and gyro together)
            imu_data = np.column_stack([accel_noisy, gyro_noisy])
            imu_noisy, imu_valid = self.add_dropouts_and_glitches(imu_data, time, 'imu')
            
            result['accel_x'] = imu_noisy[:, 0]
            result['accel_y'] = imu_noisy[:, 1]
            result['accel_z'] = imu_noisy[:, 2]
            result['gyro_x'] = imu_noisy[:, 3]
            result['gyro_y'] = imu_noisy[:, 4]
            result['gyro_z'] = imu_noisy[:, 5]
            result['imu_valid'] = imu_valid
            
            # Barometer dropouts (independent)
            altitude_noisy, baro_valid = self.add_dropouts_and_glitches(
                altitude_noisy.reshape(-1, 1), time, 'baro'
            )
            result['altitude'] = altitude_noisy.flatten()
            result['baro_valid'] = baro_valid
        else:
            result['imu_valid'] = np.ones(len(time), dtype=bool)
            result['baro_valid'] = np.ones(len(time), dtype=bool)
        
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def add_sensor_noise_to_flight(flight_data: pd.DataFrame,
                               sample_rate: float = 100.0,
                               noise_level: str = 'medium',
                               seed: int = None) -> pd.DataFrame:
    """
    Convenience function to add sensor noise to flight data
    
    Args:
        flight_data: DataFrame with columns: time, accel_x/y/z, gyro_x/y/z, altitude
        sample_rate: IMU sample rate (Hz)
        noise_level: 'low', 'medium', 'high' for different noise profiles
        seed: Random seed
        
    Returns:
        DataFrame with noisy sensor data
    """
    # Configure noise levels
    noise_configs = {
        'low': {
            'accel': AccelerometerNoise(noise_density=200e-6*9.81, bias_instability=0.01),
            'gyro': GyroscopeNoise(noise_density=0.005*np.pi/180, bias_instability=0.005*np.pi/180),
            'dropout': SensorDropoutConfig(dropout_rate=0.001, glitch_rate=0.001)
        },
        'medium': {
            'accel': AccelerometerNoise(),
            'gyro': GyroscopeNoise(),
            'dropout': SensorDropoutConfig()
        },
        'high': {
            'accel': AccelerometerNoise(noise_density=800e-6*9.81, bias_instability=0.05),
            'gyro': GyroscopeNoise(noise_density=0.02*np.pi/180, bias_instability=0.02*np.pi/180),
            'dropout': SensorDropoutConfig(dropout_rate=0.02, glitch_rate=0.01)
        }
    }
    
    config = noise_configs.get(noise_level, noise_configs['medium'])
    
    # Create simulator
    sim = SensorNoiseSimulator(
        accel_config=config['accel'],
        gyro_config=config['gyro'],
        dropout_config=config['dropout'],
        sample_rate=sample_rate,
        seed=seed
    )
    
    # Extract true values
    time = flight_data['time'].values
    accel_true = flight_data[['accel_x', 'accel_y', 'accel_z']].values
    gyro_true = flight_data[['gyro_x', 'gyro_y', 'gyro_z']].values
    altitude_true = flight_data['altitude'].values
    
    # Simulate noise
    noisy_data = sim.simulate_full_imu(accel_true, gyro_true, altitude_true, time)
    
    # Create output dataframe
    result = flight_data.copy()
    for key, values in noisy_data.items():
        result[f'{key}_noisy'] = values
    
    return result


if __name__ == "__main__":
    print("Sensor Noise Module")
    print("=" * 60)
    print("Realistic MEMS IMU noise simulation for rocket flight data")
    print()
    print("Noise sources implemented:")
    print("  ✓ Accelerometer: white noise, bias drift, temp drift")
    print("  ✓ Gyroscope: white noise, bias drift, scale errors")
    print("  ✓ Barometer: pressure noise, quantization")
    print("  ✓ GPS: position/velocity errors, multipath")
    print("  ✓ Dropouts and glitches")
    print()
    print("Example usage:")
    print("  from sensor_noise import SensorNoiseSimulator")
    print("  sim = SensorNoiseSimulator(sample_rate=100, seed=42)")
    print("  noisy_data = sim.simulate_full_imu(accel, gyro, alt, time)")
