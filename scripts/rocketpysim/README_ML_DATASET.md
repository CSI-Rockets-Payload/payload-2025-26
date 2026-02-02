# RocketPy ML Dataset Generator

Systematic generation of diverse rocket flight simulation datasets for machine learning training and testing.

## 📋 Overview

This system generates comprehensive flight simulation datasets by systematically varying:
- **Physical parameters**: mass, CG, drag, inertia
- **Thrust characteristics**: scale, duration, jitter, ripple
- **Environmental conditions**: wind, gusts, shear, turbulence, altitude
- **Launch variations**: angle offsets, rail length, misalignments
- **Anomalies**: instabilities, roll/yaw/pitch moments, failures

All variations are logged for full reproducibility and traceability.

## 🚀 Quick Start

```bash
# 1. Install requirements
pip install rocketpy numpy

# 2. Configure your dataset in set_params.py
# - Set ORK_FILE_PATH to your rocket design
# - Choose a dataset strategy
# - Adjust parameters as needed

# 3. Preview configuration
python set_params.py

# 4. Generate dataset
python rocketpy_ml_simulator.py
```

## 📁 File Structure

```
.
├── set_params.py              # Configuration file (edit this)
├── rocketpy_ml_simulator.py   # Main generator (run this)
├── base_rocket.ork            # Your OpenRocket design
└── ml_dataset/                # Output directory
    ├── dataset.csv            # Main dataset (for ML)
    ├── dataset_summary.json   # Detailed results
    ├── metadata/
    │   ├── dataset_metadata.json
    │   └── statistics.json
    └── logs/
        └── failed_simulations.json
```

## 🎯 Dataset Strategies

### Available Presets

| Strategy | Description | Simulations | Use Case |
|----------|-------------|-------------|----------|
| `minimal_test()` | Quick test | ~18 | Debugging |
| `basic_physics_variations()` | Core parameters | ~200 | Basic ML training |
| `environmental_sweep()` | Weather conditions | ~400 | Environmental robustness |
| `wind_profiles()` | Wind modeling | ~240 | Wind impact analysis |
| `launch_variations()` | Launch angles/rail | ~315 | Launch optimization |
| `thrust_anomalies()` | Motor variations | ~400 | Anomaly detection |
| `stability_anomalies()` | Instabilities | ~500 | Stability analysis |
| `comprehensive_ml_dataset()` | Full sweep | ~10,000+ | Complete ML training |
| `monte_carlo_dataset(n)` | Random sampling | n | Statistical analysis |
| `anomaly_detection_dataset()` | Normal + anomalous | ~1,000 | Anomaly detection |

### Example Configuration

```python
# In set_params.py

# Set your rocket
ORK_FILE_PATH = "my_rocket.ork"
OUTPUT_DIR = "./training_data"

# Choose strategy
PARAMETERS = DatasetStrategy.comprehensive_ml_dataset()

# Or define custom
PARAMETERS = [
    ("mass_variation_percent", [-5, 0, 5, 10]),
    ("cd_multiplier", [0.9, 1.0, 1.1]),
    ("wind_constant_x", [0, 5, 10, 15]),
    ("thrust_scale", [0.95, 1.0, 1.05]),
]
```

## 📊 Output Format

### CSV Dataset (`dataset.csv`)

Each row represents one simulation with:

**Flight Metrics** (ML Features/Targets):
- `apogee` - Maximum altitude (m)
- `apogee_time` - Time to apogee (s)
- `max_velocity` - Maximum velocity (m/s)
- `max_acceleration` - Maximum acceleration (m/s²)
- `flight_time` - Total flight duration (s)
- `impact_velocity` - Landing velocity (m/s)
- `drift_distance` - Horizontal drift (m)
- `x_impact`, `y_impact` - Landing coordinates (m)
- `stability_margin` - Static margin

**Augmentation Parameters** (ML Features):
- `mass_variation_percent` - Mass change from nominal
- `cd_multiplier` - Drag coefficient multiplier
- `thrust_scale` - Thrust scale factor
- `cg_offset_m` - CG offset (m)
- `wind_constant_x`, `wind_constant_y` - Wind components (m/s)
- `wind_turbulence` - Turbulence intensity (0-1)
- `wind_shear_rate` - Vertical wind shear
- `inclination_offset_deg` - Launch angle offset
- `heading_offset_deg` - Azimuth offset
- `roll_moment_nm`, `yaw_moment_nm` - Applied moments
- And more...

**Metadata**:
- `simulation_id` - Unique identifier
- `timestamp` - Generation time
- `success` - Boolean success flag
- `error` - Error message (if failed)

### Example Row

```csv
simulation_id,apogee,max_velocity,mass_variation_percent,cd_multiplier,wind_constant_x,...
sim_000001,1247.3,234.5,0,1.0,0,...
sim_000002,1189.2,231.8,5,1.0,5,...
sim_000003,1052.4,218.3,10,1.1,10,...
```

## 🔧 Parameter Descriptions

### Physical Variations

| Parameter | Range | Effect |
|-----------|-------|--------|
| `mass_variation_percent` | -10% to +15% | Heavier = lower apogee, slower acceleration |
| `cd_multiplier` | 0.85x to 1.20x | Higher drag = lower apogee, earlier descent |
| `cg_offset_m` | ±0.05m | Affects stability margin |
| `inertia_multiplier` | 0.9x to 1.1x | Affects rotation dynamics |

### Thrust Variations

| Parameter | Range | Effect |
|-----------|-------|--------|
| `thrust_scale` | 0.90x to 1.10x | Total impulse change |
| `burn_duration_scale` | 0.95x to 1.05x | Burn time modification |
| `thrust_jitter_percent` | 0% to 10% | Motor instability amplitude |
| `thrust_jitter_freq_hz` | 0 to 50 Hz | Jitter frequency |

### Wind Profiles

| Parameter | Range | Effect |
|-----------|-------|--------|
| `wind_constant_x/y` | 0 to 20 m/s | Steady wind components |
| `wind_gust_magnitude` | 0 to 20 m/s | Gust strength |
| `wind_gust_time` | 1 to 5 s | When gust occurs |
| `wind_shear_rate` | 0 to 0.05 | Wind change with altitude |
| `wind_turbulence` | 0 to 0.5 | Random fluctuations |

### Launch Variations

| Parameter | Range | Effect |
|-----------|-------|--------|
| `inclination_offset_deg` | ±5° | Vertical angle error |
| `heading_offset_deg` | ±10° | Azimuth error |
| `rail_length_m` | 3 to 7 m | Initial guidance length |

### Anomalies

| Parameter | Range | Effect |
|-----------|-------|--------|
| `roll_moment_nm` | 0 to 10 N·m | Asymmetric thrust/fins |
| `yaw_moment_nm` | 0 to 5 N·m | Directional instability |
| `pitch_moment_nm` | 0 to 5 N·m | Vertical instability |
| `parachute_failure` | Boolean | Recovery system failure |

## 🧪 Validation & Quality

### Automated Checks

The system validates:
- ✅ Parameter ranges are physically reasonable
- ✅ Simulations don't produce impossible results
- ✅ Failed simulations are logged separately
- ✅ All parameters are recorded for reproducibility

### Physical Plausibility

Warnings are issued for:
- Extreme mass variations (>50%)
- Unrealistic thrust scales (<0.5x or >1.5x)
- Extreme drag multipliers (<0.5x or >2.0x)

## 📈 Dataset Statistics

After generation, the system provides:

```
DATASET STATISTICS
══════════════════════════════════════════════════════════════
Total simulations: 1,234
Successful: 1,198 (97.1%)
Failed: 36 (2.9%)

Apogee (m):
  Range: [847.3, 1,456.2]
  Mean: 1,147.5 ± 124.3
  Median: 1,139.8

Max Velocity (m/s):
  Range: [198.4, 267.8]
  Mean: 234.2 ± 18.6

Drift Distance (m):
  Range: [12.3, 456.7]
  Mean: 127.4 ± 89.2
══════════════════════════════════════════════════════════════
```

## 🎓 ML Use Cases

### 1. Supervised Learning

**Predict apogee from design/environment:**
```python
features = ['mass_variation_percent', 'cd_multiplier', 'wind_constant_x', ...]
target = 'apogee'

X = dataset[features]
y = dataset[target]
model.fit(X, y)
```

### 2. Anomaly Detection

**Identify unusual flight behavior:**
```python
# Use anomaly_detection_dataset()
normal = dataset[dataset['anomaly_label'] == 'normal']
anomalous = dataset[dataset['anomaly_label'] == 'anomalous']

# Train autoencoder or isolation forest
```

### 3. Trajectory Prediction

**Multi-output regression:**
```python
targets = ['apogee', 'max_velocity', 'drift_distance', 'flight_time']
model = MultiOutputRegressor(...)
model.fit(X, y[targets])
```

### 4. Robustness Analysis

**Study sensitivity to parameters:**
```python
# Use monte_carlo_dataset()
correlation = dataset.corr()
sensitivity = dataset.groupby('wind_constant_x')['apogee'].std()
```

## 🔄 Reproducibility

### Full Parameter Logging

Every simulation logs:
- All augmentation parameters
- Timestamp
- Unique simulation ID
- Success/failure status
- Error messages (if any)

### Recreate Specific Simulations

```python
# From dataset.csv
sim_params = dataset[dataset['simulation_id'] == 'sim_000123'].iloc[0]

# Rerun with same parameters
generator.simulate_flight(sim_params.to_dict(), ...)
```

### Random Seed Control

For Monte Carlo simulations:
```python
# In set_params.py
PARAMETERS = DatasetStrategy.monte_carlo_dataset(n_samples=1000)
# Uses np.random.seed(42) for reproducibility
```

## ⚡ Performance

### Timing

- **Average**: ~2-3 seconds per simulation
- **Minimal test** (18 sims): ~1 minute
- **Comprehensive** (10,000 sims): ~6-8 hours

### Optimization Tips

1. **Start small**: Test with `minimal_test()` first
2. **Incremental generation**: Results saved continuously
3. **Parallel processing**: Future enhancement (currently serial)
4. **Resume capability**: Check existing CSV to avoid reruns

## 🐛 Troubleshooting

### Common Issues

**"RocketPy not installed"**
```bash
pip install rocketpy
```

**"Rocket file not found"**
- Check `ORK_FILE_PATH` in `set_params.py`
- Ensure `.ork` file exists

**"Too many simulations"**
- Start with fewer parameter values
- Use `minimal_test()` for debugging
- Reduce parameter combinations

**High failure rate**
- Check parameter ranges in warnings
- Review `failed_simulations.json`
- Ensure base rocket design is stable

### Validation

```bash
# Test configuration
python set_params.py

# Check for errors before full run
# Review warnings about extreme parameters
```

## 📚 References

### RocketPy
- Documentation: https://docs.rocketpy.org/
- GitHub: https://github.com/RocketPy-Team/RocketPy

### OpenRocket
- Website: https://openrocket.info/
- File format: `.ork` (XML-based)

### Machine Learning
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- TensorFlow/PyTorch for deep learning

## 📝 Citation

If you use this system in research, please cite:

```bibtex
@software{rocketpy_ml_dataset_generator,
  title={RocketPy ML Dataset Generator},
  author={Your Name},
  year={2025},
  description={Systematic parameter augmentation for rocket flight simulation datasets}
}
```

## 🤝 Contributing

To extend the system:

1. **Add new parameters**: Modify `AugmentationConfig` in `set_params.py`
2. **New strategies**: Add methods to `DatasetStrategy` class
3. **Custom metrics**: Extend `extract_flight_metrics()` in simulator
4. **Validation rules**: Add checks to `validate_parameters()`

## 📄 License

MIT License - Free to use for research and commercial applications.

## 🎯 Next Steps

1. **Generate your first dataset**:
   ```bash
   python set_params.py  # Configure
   python rocketpy_ml_simulator.py  # Generate
   ```

2. **Analyze results**:
   ```python
   import pandas as pd
   df = pd.read_csv('ml_dataset/dataset.csv')
   df.describe()
   ```

3. **Train ML models**:
   ```python
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor()
   model.fit(X_train, y_train)
   ```

4. **Iterate and improve**:
   - Add more parameters
   - Adjust ranges
   - Focus on specific scenarios

---

**Happy Dataset Generation! 🚀📊🤖**
