# 🚀 Quick Start Guide - Rocket Flight ML Dataset

Get from zero to ML-ready dataset in 5 minutes.

---

## 📋 Prerequisites

```bash
pip install rocketpy numpy pandas scikit-learn
```

**Optional (for HDF5 support):**
```bash
pip install tables
```

---

## ⚡ 3-Step Quick Start

### Step 1: Configure

Edit the top of `complete_ml_pipeline.py`:

```python
# Your rocket file
ORK_FILE = "my_rocket.ork"  # 👈 CHANGE THIS

# Dataset strategy
SIMULATION_STRATEGY = "quick_test"  # Start with quick test

# Output directory
CONFIG.output_dir = "./data"
```

### Step 2: Run

```bash
python complete_ml_pipeline.py
```

Or for a quick test:
```bash
python complete_ml_pipeline.py --mode quick
```

### Step 3: Load & Train

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv('data/rocket_flight_ml_train.csv')
test = pd.read_csv('data/rocket_flight_ml_test.csv')

# Train model
features = ['mass_variation_percent', 'cd_multiplier', 'wind_constant_x']
target = 'apogee'

model = RandomForestRegressor()
model.fit(train[features], train[target])

# Predict
predictions = model.predict(test[features])
```

**Done!** 🎉

---

## 📁 Output Structure

After running the pipeline:

```
your_project/
├── complete_ml_pipeline.py          # Main pipeline
├── rocketpy_ml_dataset_generator.py # Simulation generator
├── preprocess_dataset.py            # Preprocessing module
├── sensor_noise.py                  # Sensor noise models
├── my_rocket.ork                    # Your rocket design
│
├── raw_simulations/                 # Stage 1 output
│   ├── dataset.csv
│   ├── results_full.json
│   └── metadata/
│
└── data/                            # Stage 2 output (ML-ready)
    ├── rocket_flight_ml_train.csv   # Training set
    ├── rocket_flight_ml_val.csv     # Validation set
    ├── rocket_flight_ml_test.csv    # Test set
    ├── rocket_flight_ml_metadata.json
    ├── rocket_flight_ml_scaler.pkl
    └── preprocessing_report.txt
```

---

## 🎯 Dataset Strategies

Choose one by setting `SIMULATION_STRATEGY`:

| Strategy | Simulations | Time | Use Case |
|----------|-------------|------|----------|
| `quick_test` | ~18 | 1 min | Debugging, testing |
| `basic_physics` | ~200 | 10 min | Basic ML training |
| `environmental_sweep` | ~400 | 20 min | Weather sensitivity |
| `wind_profiles` | ~240 | 12 min | Wind analysis |
| `comprehensive` | 10,000+ | 6-8 hrs | Full production dataset |
| `monte_carlo` | Custom | Variable | Statistical analysis |
| `anomaly_detection` | ~1,000 | 45 min | Anomaly detection |

---

## 🔧 Configuration Options

### Simulation Parameters

```python
# In complete_ml_pipeline.py

ORK_FILE = "my_rocket.ork"
SIMULATION_STRATEGY = "basic_physics"
RAW_DATA_DIR = "./raw_simulations"
```

### Preprocessing Options

```python
CONFIG.train_ratio = 0.70           # 70% training
CONFIG.val_ratio = 0.15             # 15% validation
CONFIG.test_ratio = 0.15            # 15% test

CONFIG.add_sensor_noise = True      # Add realistic sensor noise
CONFIG.sensor_noise_level = 'medium' # 'low', 'medium', 'high'

CONFIG.augment_training_data = True  # Create multiple noisy versions
CONFIG.augmentation_factor = 3       # 3x augmentation

CONFIG.normalize_features = True     # Normalize features
CONFIG.stratify_by = 'apogee'       # Balance splits by apogee
```

---

## 📊 Data Columns

### Key Features (Inputs)
- `mass_variation_percent` - Mass change from nominal
- `cd_multiplier` - Drag coefficient multiplier
- `thrust_scale` - Thrust scale factor
- `wind_constant_x`, `wind_constant_y` - Wind components
- `inclination_offset_deg` - Launch angle error
- `cg_offset_m` - Center of gravity offset

### Targets (Outputs)
- `apogee` - Maximum altitude (m)
- `max_velocity` - Peak velocity (m/s)
- `flight_time` - Total duration (s)
- `drift_distance` - Horizontal drift (m)
- `impact_velocity` - Landing speed (m/s)

### Metadata
- `simulation_id` - Unique ID
- `success` - Boolean success flag
- `augmentation_id` - 0=original, 1+=augmented

---

## 🎓 Common ML Tasks

### Task 1: Predict Apogee

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('data/rocket_flight_ml_train.csv')
test = pd.read_csv('data/rocket_flight_ml_test.csv')

features = ['mass_variation_percent', 'cd_multiplier', 'thrust_scale',
            'wind_constant_x', 'inclination_offset_deg']
target = 'apogee'

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[features], train[target])

predictions = model.predict(test[features])
```

### Task 2: Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

train = pd.read_csv('data/rocket_flight_ml_train.csv')

features = ['apogee', 'max_velocity', 'flight_time', 'drift_distance']
X = train[features]

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

anomalies = clf.predict(X)
```

### Task 3: Multi-Output Prediction

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

targets = ['apogee', 'max_velocity', 'drift_distance', 'flight_time']
features = ['mass_variation_percent', 'cd_multiplier', 'wind_constant_x']

model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(train[features], train[targets])
```

---

## 🐛 Troubleshooting

### Issue: "RocketPy not installed"
```bash
pip install rocketpy
```

### Issue: "Rocket file not found"
- Check `ORK_FILE` path is correct
- Make sure `.ork` file exists
- Use absolute path if needed

### Issue: "Too many simulations"
- Start with `quick_test` strategy
- Reduce parameter ranges
- Use `--mode quick` flag

### Issue: "Out of memory"
- Reduce `augmentation_factor`
- Use `save_hdf5` instead of CSV
- Process in batches

### Issue: "Dataset too small"
- Increase to `comprehensive` strategy
- Increase `augmentation_factor`
- Use `monte_carlo` with more samples

---

## 📈 Performance Tips

### Faster Generation
```python
# Use simpler strategies
SIMULATION_STRATEGY = "quick_test"

# Reduce augmentation
CONFIG.augmentation_factor = 1

# Disable sensor noise during testing
CONFIG.add_sensor_noise = False
```

### Larger Datasets
```python
# Use comprehensive strategy
SIMULATION_STRATEGY = "comprehensive"

# Increase augmentation
CONFIG.augmentation_factor = 5

# Add more parameter variations
# (edit DatasetStrategies in rocketpy_ml_dataset_generator.py)
```

### Better Quality
```python
# Add realistic sensor noise
CONFIG.add_sensor_noise = True
CONFIG.sensor_noise_level = 'high'

# Use stratification
CONFIG.stratify_by = 'apogee'

# Normalize features
CONFIG.normalize_features = True
```

---

## 🔄 Reproducibility

All random operations use **seed 42** for reproducibility:

```python
CONFIG.random_seed = 42  # Fixed seed
```

To recreate exact same dataset:
1. Use same `.ork` file
2. Use same configuration
3. Use same random seed
4. Run pipeline again

---

## 📚 Next Steps

1. **Explore the data:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/rocket_flight_ml_train.csv')
   df.describe()
   df.hist(figsize=(15, 10))
   ```

2. **Read full documentation:**
   - `DATA_DOCUMENTATION.md` - Complete specs
   - `README_ML_DATASET.md` - Detailed guide

3. **Try different strategies:**
   - Start with `quick_test`
   - Move to `basic_physics`
   - Scale to `comprehensive`

4. **Train your models:**
   - Regression (predict apogee)
   - Classification (anomaly detection)
   - Time-series (if trajectory data)

5. **Validate results:**
   - Use validation set for hyperparameters
   - Use test set for final evaluation
   - Report metrics on test set only

---

## 💡 Tips & Best Practices

### DO:
✅ Start with `quick_test` to verify pipeline  
✅ Use validation set for model selection  
✅ Keep test set untouched until final eval  
✅ Document your model choices  
✅ Save trained models and scalers  

### DON'T:
❌ Don't use test set during development  
❌ Don't fit scalers on test data  
❌ Don't cherry-pick best test results  
❌ Don't ignore failed simulations  
❌ Don't forget to set random seed  

---

## 🆘 Getting Help

**Check logs:**
```bash
# Simulation logs
cat raw_simulations/metadata/info.json

# Preprocessing logs
cat data/preprocessing_report.txt
```

**Verify data:**
```python
import pandas as pd

# Check for NaN
df = pd.read_csv('data/rocket_flight_ml_train.csv')
print(df.isnull().sum())

# Check distributions
print(df.describe())

# Check splits
train = pd.read_csv('data/rocket_flight_ml_train.csv')
val = pd.read_csv('data/rocket_flight_ml_val.csv')
test = pd.read_csv('data/rocket_flight_ml_test.csv')
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

---

## 🎉 You're Ready!

You now have a complete ML-ready dataset with:
- ✅ Systematic parameter variations
- ✅ Realistic sensor noise
- ✅ Proper train/val/test splits
- ✅ Normalized features
- ✅ Full reproducibility

**Happy modeling! 🚀🤖**

---

*For detailed documentation, see DATA_DOCUMENTATION.md*
