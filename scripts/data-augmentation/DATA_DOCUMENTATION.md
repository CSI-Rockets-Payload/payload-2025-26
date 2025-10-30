# Rocket Flight ML Dataset Documentation

**Version:** 1.0  
**Date:** 2025  
**Purpose:** Machine learning training dataset for rocket flight prediction and anomaly detection

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Generation Pipeline](#dataset-generation-pipeline)
3. [Parameter Augmentation](#parameter-augmentation)
4. [Sensor Noise Models](#sensor-noise-models)
5. [Train/Val/Test Splits](#trainvaltest-splits)
6. [Data Format](#data-format)
7. [Reproducibility](#reproducibility)
8. [Usage Examples](#usage-examples)
9. [Assumptions & Limitations](#assumptions--limitations)

---

## Overview

This dataset contains rocket flight simulation data generated using **RocketPy** from OpenRocket (.ork) design files. The dataset includes:

- **Parametric variations** of physical, environmental, and operational parameters
- **Realistic sensor noise** models (IMU, GPS, barometer)
- **Data augmentation** with multiple noisy versions per sample
- **Stratified splits** for balanced train/validation/test sets
- **Comprehensive metadata** for full reproducibility

### Dataset Statistics

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | Variable | 70% |
| Validation | Variable | 15% |
| Test | Variable | 15% |

**Total parameters varied:** 10+ (mass, drag, thrust, wind, CG, etc.)  
**Augmentation factor:** 3x (configurable)  
**Random seed:** 42 (reproducible)

---

## Dataset Generation Pipeline

The complete pipeline consists of 4 stages:

### Stage 1: RocketPy Simulation

```
OpenRocket .ork file
    ↓
RocketPy simulation engine
    ↓
Parameter grid sweep
    ↓
Raw flight metrics (CSV)
```

**Parameters varied:**
- Physical: mass (±15%), drag coefficient (0.85-1.2x), CG offset (±5cm)
- Thrust: scale (±10%), jitter (0-5%), duration (±5%)
- Environmental: wind (0-20 m/s), altitude (0-2000m), temperature (±15°C)
- Launch: angle offsets (±3°), rail length (3-7m)
- Anomalies: roll/yaw moments, instabilities

**Output:** `raw_simulations/dataset.csv`

### Stage 2: Sensor Noise Injection

```
Raw simulation data
    ↓
IMU noise model
    ↓
GPS noise model
    ↓
Barometer noise model
    ↓
Dropout & glitch simulation
    ↓
Augmented dataset (Nx augmentation_factor)
```

**Noise models applied:**
- Accelerometer: white noise, bias drift, temperature effects
- Gyroscope: angle random walk, bias instability
- Barometer: pressure noise, quantization
- GPS: position errors, multipath, update rate
- Dropouts: random sensor failures (1% rate)

**Output:** Augmented dataset with realistic sensor noise

### Stage 3: Train/Val/Test Split

```
Augmented dataset
    ↓
Stratified sampling (by apogee bins)
    ↓
70% Train | 15% Val | 15% Test
    ↓
Independent datasets
```

**Stratification:** Ensures balanced representation across apogee ranges

### Stage 4: Feature Normalization

```
Train/Val/Test splits
    ↓
Fit scaler on training data
    ↓
Transform all splits
    ↓
Save scaler for inference
    ↓
ML-ready datasets
```

**Normalization:** StandardScaler (zero mean, unit variance)

---

## Parameter Augmentation

### Physical Parameters

| Parameter | Range | Distribution | Effect |
|-----------|-------|--------------|--------|
| `mass_variation_percent` | -10% to +15% | Uniform | Affects acceleration, apogee |
| `cd_multiplier` | 0.85 to 1.20 | Uniform | Drag coefficient scaling |
| `cg_offset_m` | ±0.05 m | Uniform | Center of gravity shift |
| `inertia_multiplier` | 0.9 to 1.1 | Uniform | Moment of inertia scaling |

**Assumptions:**
- Mass variations simulate payload changes, propellant loading errors
- CG offsets represent manufacturing tolerances, payload positioning
- Drag variations account for surface finish, protrusions, damage

### Thrust Variations

| Parameter | Range | Distribution | Effect |
|-----------|-------|--------------|--------|
| `thrust_scale` | 0.90 to 1.10 | Uniform | Total impulse variation |
| `burn_duration_scale` | 0.95 to 1.05 | Uniform | Burn time modification |
| `thrust_jitter_percent` | 0% to 5% | Uniform | Motor instability amplitude |
| `thrust_jitter_freq_hz` | 0 to 50 Hz | Discrete | Jitter frequency |

**Assumptions:**
- Thrust variations simulate motor batch differences
- Jitter represents combustion instabilities (CATO precursors)
- Based on commercial motor tolerances (±10% impulse typical)

### Environmental Conditions

| Parameter | Range | Distribution | Effect |
|-----------|-------|--------------|--------|
| `wind_constant_x` | 0 to 20 m/s | Uniform | East-West wind |
| `wind_constant_y` | 0 to 15 m/s | Uniform | North-South wind |
| `wind_turbulence` | 0 to 0.5 | Uniform | Random fluctuations |
| `wind_shear_rate` | 0 to 0.05 | Uniform | Wind change with altitude |
| `launch_altitude_m` | 0 to 2000 m | Discrete | Launch site elevation |
| `temperature_offset_k` | ±15 K | Uniform | Temperature variation |

**Assumptions:**
- Wind profiles based on typical surface conditions
- Turbulence intensity from atmospheric boundary layer models
- Temperature affects air density (simple ideal gas model)

### Launch Variations

| Parameter | Range | Distribution | Effect |
|-----------|-------|--------------|--------|
| `inclination_offset_deg` | ±3° | Uniform | Vertical alignment error |
| `heading_offset_deg` | ±5° | Uniform | Azimuth misalignment |
| `rail_length_m` | 3.0 to 7.0 m | Discrete | Initial guidance length |

**Assumptions:**
- Angle offsets represent launch tower misalignment
- Rail length variations simulate different launch facilities
- Errors uncorrelated (independent random variations)

### Anomalies

| Parameter | Range | Distribution | Effect |
|-----------|-------|--------------|--------|
| `roll_moment_nm` | 0 to 10 N·m | Discrete | Asymmetric thrust/fin damage |
| `yaw_moment_nm` | 0 to 5 N·m | Discrete | Directional instability |
| `pitch_moment_nm` | 0 to 5 N·m | Discrete | Vertical instability |
| `parachute_failure` | Boolean | Bernoulli | Recovery system failure |

**Assumptions:**
- Moments simulate fin damage, asymmetric thrust, aerodynamic anomalies
- Magnitudes based on typical fin forces at max Q
- Failure modes are binary (works/doesn't work)

---

## Sensor Noise Models

All sensor noise models are based on **typical MEMS IMU specifications** (e.g., MPU6050, BNO055, BMI088).

### Accelerometer Noise

**Sensor:** 16-bit MEMS accelerometer, ±16g range

| Noise Source | Value | Units | Description |
|--------------|-------|-------|-------------|
| White noise density | 400 µg/√Hz | m/s²/√Hz | High-frequency noise |
| Bias instability | 0.02 | m/s² | ~2 mg slow drift |
| Bias random walk | 0.001 | m/s²/√s | Bias drift rate |
| Initial bias | σ=0.05 | m/s² | ~5 mg offset |
| Temperature drift | 0.01 | m/s²/°C | Thermal sensitivity |
| Scale factor error | 2% | % | Gain error |
| Saturation | ±160 | m/s² | ±16g limit |
| Quantization | 16 bits | - | ADC resolution |

**Noise Model:**
```
a_measured = a_true × (1 + scale_error) + bias + bias_drift + white_noise + temp_drift
```

**Assumptions:**
- Bias drift modeled as random walk
- Temperature varies ±5°C sinusoidally over 60s
- White noise scaled by √sample_rate
- Saturation is hard-clipped

### Gyroscope Noise

**Sensor:** 16-bit MEMS gyroscope, ±2000°/s range

| Noise Source | Value | Units | Description |
|--------------|-------|-------|-------------|
| White noise density | 0.01 °/s/√Hz | rad/s/√Hz | Angular random walk |
| Bias instability | 0.01 °/s | rad/s | ~36°/hr drift |
| Bias random walk | 0.0001 °/s/√s | rad/s/√s | Bias drift rate |
| Initial bias | σ=0.1 °/s | rad/s | Startup offset |
| Temperature drift | 0.02 °/s/°C | rad/s/°C | Thermal sensitivity |
| Scale factor error | 2% | % | Gain error |
| Saturation | ±2000 °/s | rad/s | Full scale range |
| Quantization | 16 bits | - | ADC resolution |

**Noise Model:**
```
ω_measured = ω_true × (1 + scale_error) + bias + bias_drift + white_noise + temp_drift
```

**Assumptions:**
- Same temperature profile as accelerometer
- Bias drift is correlated across axes (common mode)
- Quantization uses least significant bit (LSB) rounding

### Barometer/Altimeter Noise

**Sensor:** High-precision barometer (MS5611, BMP280 class)

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| Pressure noise | 2.0 Pa | Pa | ~0.17 m RMS |
| Altitude resolution | 0.01 m | m | Quantization |
| Bias drift rate | 0.1 Pa/s | Pa/s | Slow drift |
| Temperature sensitivity | 0.5 Pa/°C | Pa/°C | Thermal effect |

**Noise Model:**
```
h_measured = h_true + pressure_noise/12.0 + bias_drift + temp_effect
```

**Conversion:** 1 Pa ≈ 0.083 m altitude (at sea level)

**Assumptions:**
- Standard atmosphere model (exponential pressure decay)
- Bias drift is cumulative (random walk)
- Temperature effect is linear

### GPS Noise

**Sensor:** Consumer GPS receiver (u-blox M8 class)

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| Horizontal accuracy (CEP) | 2.5 m | m | Circular error probable |
| Vertical accuracy | 5.0 m | m | Altitude error |
| Velocity accuracy | 0.1 m/s | m/s | Speed error |
| Update rate | 5 Hz | Hz | Position updates |
| Multipath std | 1.0 m | m | Reflection errors |

**Noise Model:**
```
pos_measured = pos_true + gps_noise + multipath × sin(2πt/10)
vel_measured = vel_true + vel_noise
```

**Assumptions:**
- Updates at 5 Hz, values held between updates (zero-order hold)
- Multipath is sinusoidal (10s period, simplified)
- Horizontal/vertical errors independent
- No SA (Selective Availability), ionospheric correction enabled

### Sensor Dropouts & Glitches

| Parameter | Value | Description |
|-----------|-------|-------------|
| Dropout rate | 1% per second | Probability of sensor failure |
| Dropout duration | 10-500 ms | Random uniform |
| Glitch rate | 0.5% per second | Probability of spike |
| Glitch magnitude | 10× noise std | Spike amplitude |

**Dropout Model:**
- During dropout: `measurement = NaN` or hold last value
- Validity flag set to `False`
- IMU dropouts affect accel + gyro together
- Barometer dropouts independent

**Glitch Model:**
- Random large spike (±10σ)
- Single sample affected
- Independent across axes

**Assumptions:**
- Dropout durations from electrical transients, bus errors
- Glitches from EMI, static discharge
- No permanent sensor failures (all recover)

---

## Train/Val/Test Splits

### Splitting Strategy

**Method:** Stratified random split

```python
train_ratio = 0.70  # 70% training
val_ratio = 0.15    # 15% validation
test_ratio = 0.15   # 15% test
random_seed = 42    # Reproducible splits
```

**Stratification variable:** `apogee` (binned into 5 quantiles)

**Rationale:**
- Ensures balanced representation of low/medium/high apogee flights
- Prevents train/test distribution mismatch
- Allows reliable performance evaluation across flight regimes

### Split Characteristics

| Split | Purpose | Usage | Characteristics |
|-------|---------|-------|-----------------|
| **Training** | Model fitting | Fit model parameters, hyperparameters | Largest set, includes augmented data |
| **Validation** | Hyperparameter tuning | Select best model, early stopping | Independent from training |
| **Test** | Final evaluation | Report final metrics, generalization | Never used during development |

### Data Leakage Prevention

**Measures taken:**
1. ✅ Augmentation AFTER splitting (train-only augmentation)
2. ✅ Scaler fit on training set only
3. ✅ Stratification on original samples (before augmentation)
4. ✅ No test set access until final evaluation
5. ✅ Temporal independence (simulations independent)

### Cross-Validation Alternative

For smaller datasets, k-fold cross-validation can be used:

```python
# Instead of fixed split:
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Note:** Not used by default to maintain simple train/val/test paradigm

---

## Data Format

### CSV Structure

```csv
simulation_id,timestamp,success,apogee,max_velocity,mass_variation_percent,...
sim_000001,2025-01-15T10:23:45,True,1247.3,234.5,0,...
sim_000002,2025-01-15T10:23:47,True,1189.2,231.8,5,...
...
```

### Column Categories

#### Metadata Columns
- `simulation_id`: Unique identifier (string)
- `timestamp`: Generation timestamp (ISO 8601)
- `success`: Boolean flag (True if simulation succeeded)
- `error`: Error message (if success=False)
- `augmentation_id`: 0=original, 1+=augmented versions

#### Flight Performance Metrics (Targets)
- `apogee`: Maximum altitude (m)
- `apogee_time`: Time to apogee (s)
- `max_velocity`: Maximum velocity (m/s)
- `max_acceleration`: Maximum acceleration (m/s²)
- `max_mach`: Maximum Mach number
- `flight_time`: Total flight duration (s)
- `impact_velocity`: Landing velocity (m/s)
- `x_impact`, `y_impact`: Landing coordinates (m)
- `drift_distance`: Horizontal drift (m)
- `stability_margin`: Static margin
- `out_of_rail_velocity`: Velocity leaving launch rail (m/s)

#### Augmentation Parameters (Features)
- `mass_variation_percent`: Mass change (%)
- `cd_multiplier`: Drag coefficient multiplier
- `thrust_scale`: Thrust scale factor
- `cg_offset_m`: CG offset (m)
- `inertia_multiplier`: Inertia multiplier
- `burn_duration_scale`: Burn time scale
- `thrust_jitter_percent`: Jitter amplitude (%)
- `thrust_jitter_freq_hz`: Jitter frequency (Hz)
- `wind_constant_x`, `wind_constant_y`: Wind components (m/s)
- `wind_turbulence`: Turbulence intensity (0-1)
- `wind_shear_rate`: Wind shear (m/s per m altitude)
- `inclination_offset_deg`: Launch angle offset (deg)
- `heading_offset_deg`: Azimuth offset (deg)
- `rail_length_m`: Launch rail length (m)
- `launch_altitude_m`: Launch site elevation (m)
- `launch_latitude`: Launch latitude (deg)
- `temperature_offset_k`: Temperature offset (K)
- `pressure_multiplier`: Pressure multiplier
- `roll_moment_nm`, `yaw_moment_nm`, `pitch_moment_nm`: Applied moments (N·m)
- `parachute_failure`: Boolean failure flag
- `parachute_delay_s`: Deployment delay (s)

#### Sensor Noise Indicators (if applicable)
- `imu_valid`: Boolean array of IMU validity
- `baro_valid`: Boolean array of barometer validity
- `accel_x_noisy`, `accel_y_noisy`, `accel_z_noisy`: Noisy accelerometer (m/s²)
- `gyro_x_noisy`, `gyro_y_noisy`, `gyro_z_noisy`: Noisy gyroscope (rad/s)
- `altitude_noisy`: Noisy barometer (m)

### File Formats

| Format | Files | Use Case | Pros | Cons |
|--------|-------|----------|------|------|
| **CSV** | `*_train.csv`, `*_val.csv`, `*_test.csv` | Inspection, simple ML | Human-readable, universal | Large file size |
| **HDF5** | `*_train.h5`, `*_val.h5`, `*_test.h5` | Large datasets, fast loading | Compressed, fast | Binary format |
| **NumPy** | `*_train_features.npy`, etc. | Deep learning pipelines | Very fast, memory-mapped | No column names |
| **Pickle** | `*_scaler.pkl` | Model deployment | Stores Python objects | Not portable |
| **JSON** | `*_metadata.json` | Metadata, documentation | Structured, readable | Not for large data |

### Loading Examples

```python
import pandas as pd
import numpy as np
import pickle

# Load CSV
train_df = pd.read_csv('data/rocket_flight_ml_train.csv')

# Load HDF5
train_df = pd.read_hdf('data/rocket_flight_ml_train.h5', key='data')

# Load NumPy
X_train = np.load('data/rocket_flight_ml_train_features.npy')

# Load scaler
with open('data/rocket_flight_ml_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

---

## Reproducibility

### Random Seed Control

**All random operations use seed 42:**
- NumPy: `np.random.seed(42)`
- Train/test split: `random_state=42`
- Augmentation: Seed set before each stage

### Parameter Logging

**Every simulation logs:**
1. All input parameters (rocket, environment, launch)
2. Augmentation settings (noise levels, factors)
3. Preprocessing steps (normalization, scaling)
4. Software versions (RocketPy, Python, libraries)
5. Timestamps (generation date/time)

### Metadata Files

```
data/
├── rocket_flight_ml_metadata.json      # Dataset-level metadata
├── rocket_flight_ml_scaler.pkl         # Normalization scaler
├── preprocessing_report.txt            # Human-readable summary
└── raw_simulations/
    └── dataset_metadata.json           # Simulation parameters
```

### Recreating Exact Dataset

```bash
# 1. Use exact same .ork file
# 2. Use same random seed (42)
# 3. Run pipeline with same configuration
python complete_ml_pipeline.py --mode normal
```

**Checksum verification:**
```bash
md5sum data/rocket_flight_ml_train.csv
# Should match provided checksum
```

---

## Usage Examples

### Example 1: Train Regression Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train = pd.read_csv('data/rocket_flight_ml_train.csv')
test = pd.read_csv('data/rocket_flight_ml_test.csv')

# Define features and target
feature_cols = ['mass_variation_percent', 'cd_multiplier', 'thrust_scale',
                'wind_constant_x', 'wind_constant_y', 'inclination_offset_deg']
target_col = 'apogee'

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f} m")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### Example 2: Multi-Output Prediction

```python
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load data
train = pd.read_csv('data/rocket_flight_ml_train.csv')

# Multiple targets
targets = ['apogee', 'max_velocity', 'drift_distance', 'flight_time']
features = [c for c in train.columns if c not in targets + ['simulation_id', 'timestamp']]

X_train = train[features]
y_train = train[targets]

# Multi-output model
model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(X_train, y_train)
```

### Example 3: Anomaly Detection

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
train = pd.read_csv('data/rocket_flight_ml_train.csv')

# Use only successful flights for normal behavior
normal_flights = train[train['success'] == True]

# Train anomaly detector
features = ['apogee', 'max_velocity', 'flight_time', 'drift_distance']
X = normal_flights[features]

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# Predict anomalies
predictions = clf.predict(X)
anomalies = normal_flights[predictions == -1]
print(f"Detected {len(anomalies)} anomalous flights")
```

### Example 4: Time Series Prediction (if time-series data available)

```python
import pandas as pd
import numpy as np
from tensorflow import keras

# Load time-series data (hypothetical)
# Assuming we have acceleration vs time for each flight
train = pd.read_csv('data/rocket_flight_ml_train_timeseries.csv')

# Prepare sequences
def create_sequences(data, seq_length=100):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train['acceleration'].values)

# LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(100, 1)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## Assumptions & Limitations

### Modeling Assumptions

1. **Rigid Body Dynamics**
   - Rocket treated as rigid body (no structural flexibility)
   - No propellant slosh effects
   - No aeroelastic effects

2. **Atmospheric Model**
   - Standard atmosphere or simple exponential model
   - No weather systems (fronts, inversions)
   - Uniform wind (no gusts except parameterized)

3. **Aerodynamics**
   - Coefficients constant or slowly varying with Mach
   - No transonic effects beyond drag curve
   - No vortex shedding, flow separation details

4. **Motor Performance**
   - Thrust curve from manufacturer data or simple model
   - No combustion instabilities (except parameterized jitter)
   - No nozzle erosion, pressure variations

5. **Sensor Models**
   - Based on commercial-off-the-shelf (COTS) MEMS sensors
   - Simplified noise models (no complex correlations)
   - No sensor aging effects
   - No magnetic interference for magnetometers

### Known Limitations

1. **Simulation Fidelity**
   - RocketPy is 6-DOF but some effects simplified
   - No CFD-level aerodynamics
   - Monte Carlo wind better than simple constant wind

2. **Parameter Ranges**
   - Limited to ±15% mass, ±20% thrust (typical ranges)
   - May not cover extreme anomalies (explosion, tumbling)
   - Wind limited to surface layer (no jet stream)

3. **Sensor Noise**
   - Dropout model simplified (no permanent failures)
   - No bus errors, communication protocol issues
   - GPS multipath simplified (no building reflections)

4. **Data Augmentation**
   - Assumes independence (noise added independently)
   - No temporal correlations beyond single flight
   - Augmentation may overfit to noise model

5. **Generalization**
   - Trained on specific rocket design (.ork file)
   - May not generalize to very different rockets
   - Weather conditions limited to training range

### Recommended Use Cases

**✅ Good for:**
- Apogee prediction from pre-launch parameters
- Trajectory optimization
- Launch condition sensitivity analysis
- Anomaly detection (motor failure, wind gusts)
- Control system testing (simulated sensor noise)

**⚠️ Use with caution for:**
- Transonic/supersonic regime (Mach > 0.8)
- Very high altitude (> 5 km)
- Complex wind patterns (mountains, urban)
- Long-duration flights (> 5 min)

**❌ Not suitable for:**
- Orbital mechanics
- Hypersonic flight
- Multi-stage complex vehicles
- Detailed combustion modeling

### Future Improvements

Potential enhancements to dataset:

1. **Time-series data**: Include full trajectory (not just summary statistics)
2. **Video/image data**: Synthetic camera views for vision ML
3. **Weather data**: Real atmospheric profiles from soundings
4. **Hardware-in-loop**: Real sensor data from test stands
5. **CFD validation**: High-fidelity drag curves
6. **Extended anomalies**: More failure modes (fin loss, parachute tangle)
7. **Multi-rocket**: Multiple rocket designs in one dataset

---

## Citation

If you use this dataset in research, please cite:

```bibtex
@dataset{rocket_flight_ml_2025,
  title={Rocket Flight ML Dataset with Sensor Noise Augmentation},
  author={Your Name},
  year={2025},
  publisher={Your Organization},
  version={1.0},
  url={https://github.com/yourrepo/rocket-ml-dataset}
}
```

---

## Contact & Support

For questions, issues, or contributions:

- **GitHub Issues:** https://github.com/yourrepo/issues
- **Email:** your.email@example.com
- **Documentation:** Full docs at https://yourrepo.github.io/rocket-ml-dataset

---

**Last Updated:** 2025-01-15  
**Dataset Version:** 1.0  
**License:** MIT / Apache 2.0 / CC BY 4.0 (choose appropriate)
