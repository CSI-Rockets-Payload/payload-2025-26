#!/usr/bin/env python3
"""
Complete ML Dataset Generation Pipeline
End-to-end pipeline from .ork file to ML-ready datasets.

Pipeline stages:
1. RocketPy simulations with parameter variations
2. Sensor noise injection (IMU, GPS, barometer)
3. Train/validation/test splits
4. Feature normalization
5. Export to multiple formats

Usage:
    python complete_ml_pipeline.py
    
Or configure at the top and run.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add the ../rocketpysim folder to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "rocketpysim"))

import rocketpy_ml_simulator as sim_gen


# ╔════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                              ║
# ╚════════════════════════════════════════════════════════════════════╝

# ============================================================
# STEP 1: SIMULATION CONFIGURATION
# ============================================================

# Your rocket file
ORK_FILE = "../rocketpysim/template_rocket.ork"

# Dataset generation strategy
SIMULATION_STRATEGY = "basic_physics"  # See rocketpy_ml_dataset_generator.py for options

# Simulation output directory
RAW_DATA_DIR = "./raw_simulations"


# ============================================================
# STEP 2: PREPROCESSING CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    
    # --- Data Splits ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # --- Reproducibility ---
    random_seed: int = 42
    
    # --- Sensor Noise ---
    add_sensor_noise: bool = True
    sensor_noise_level: str = 'medium'  # 'low', 'medium', 'high'
    imu_sample_rate: float = 100.0
    
    # --- Data Augmentation ---
    augment_training_data: bool = True
    augmentation_factor: int = 3  # Number of noisy versions per sample
    
    # --- Normalization ---
    normalize_features: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # --- Stratification ---
    stratify_by: Optional[str] = 'apogee'  # Column to balance splits
    
    # --- Output ---
    output_dir: str = "./data"
    dataset_name: str = "rocket_flight_ml"
    save_csv: bool = True
    save_hdf5: bool = True
    save_numpy: bool = False


# Create configuration
CONFIG = PipelineConfig()


# ╔════════════════════════════════════════════════════════════════════╗
# ║                         PIPELINE EXECUTION                         ║
# ╚════════════════════════════════════════════════════════════════════╝

def run_complete_pipeline():
    """Execute complete ML dataset generation pipeline"""
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          Complete ML Dataset Generation Pipeline                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    # Validate configuration
    print("📋 Configuration:")
    print(f"   Rocket file: {ORK_FILE}")
    print(f"   Strategy: {SIMULATION_STRATEGY}")
    print(f"   Random seed: {CONFIG.random_seed}")
    print(f"   Output: {CONFIG.output_dir}")
    print()
    
    if not Path(ORK_FILE).exists():
        print(f"❌ Rocket file not found: {ORK_FILE}")
        print("   Please update ORK_FILE at the top of this script.")
        return 1
    
    # ========================================
    # STAGE 1: Generate Simulations
    # ========================================
    print("="*70)
    print("STAGE 1: RocketPy Simulation Generation")
    print("="*70)
    
    try:
        print("\n🚀 Running RocketPy simulations...")
        print(f"   This may take a while depending on the strategy...\n")
        
        # Set configuration
        sim_gen.ORK_FILE = ORK_FILE
        sim_gen.OUTPUT_DIR = RAW_DATA_DIR
        sim_gen.STRATEGY = SIMULATION_STRATEGY
        
        # Run simulations
        generator = sim_gen.RocketPyMLGenerator(ORK_FILE, RAW_DATA_DIR)
        
        # Get strategy
        strategies = sim_gen.DatasetStrategies()
        strategy_map = {
            'quick_test': strategies.quick_test,
            'basic_physics': strategies.basic_physics,
            'environmental_sweep': strategies.environmental_sweep,
            'wind_profiles': strategies.wind_profiles,
            'launch_variations': strategies.launch_variations,
            'thrust_anomalies': strategies.thrust_anomalies,
            'stability_anomalies': strategies.stability_anomalies,
            'comprehensive': strategies.comprehensive,
            'monte_carlo': lambda: strategies.monte_carlo(100),
            'anomaly_detection': strategies.anomaly_detection,
        }
        
        if SIMULATION_STRATEGY not in strategy_map:
            print(f"❌ Invalid strategy: {SIMULATION_STRATEGY}")
            return 1
        
        parameters = strategy_map[SIMULATION_STRATEGY]()
        generator.generate_dataset(parameters)
        
        raw_csv = Path(RAW_DATA_DIR) / "dataset.csv"
        
        print(f"\n✅ Stage 1 complete!")
        print(f"   Raw data: {raw_csv}")
        
    except Exception as e:
        print(f"\n❌ Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================
    # STAGE 2: Preprocess & Split Data
    # ========================================
    print("\n" + "="*70)
    print("STAGE 2: Data Preprocessing & Train/Test Split")
    print("="*70)
    
    try:
        print("\n🔧 Preprocessing data...")
        
        # Import preprocessing module
        import preprocess_dataset as preproc
        
        # Create config
        preproc_config = preproc.PreprocessConfig()
        preproc_config.train_ratio = CONFIG.train_ratio
        preproc_config.val_ratio = CONFIG.val_ratio
        preproc_config.test_ratio = CONFIG.test_ratio
        preproc_config.random_seed = CONFIG.random_seed
        preproc_config.add_sensor_noise = CONFIG.add_sensor_noise
        preproc_config.sensor_noise_level = CONFIG.sensor_noise_level
        preproc_config.imu_sample_rate = CONFIG.imu_sample_rate
        preproc_config.augment_training_data = CONFIG.augment_training_data
        preproc_config.augmentation_factor = CONFIG.augmentation_factor
        preproc_config.normalize_features = CONFIG.normalize_features
        preproc_config.normalization_method = CONFIG.normalization_method
        preproc_config.save_csv = CONFIG.save_csv
        preproc_config.save_hdf5 = CONFIG.save_hdf5
        preproc_config.save_numpy = CONFIG.save_numpy
        preproc_config.stratify_by = CONFIG.stratify_by
        
        # Run preprocessing
        preproc.preprocess_pipeline(
            str(raw_csv),
            CONFIG.output_dir,
            preproc_config
        )
        
        print(f"\n✅ Stage 2 complete!")
        print(f"   Processed data: {CONFIG.output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\n📁 Output Structure:")
    print(f"   {CONFIG.output_dir}/")
    print(f"   ├── {CONFIG.dataset_name}_train.csv")
    print(f"   ├── {CONFIG.dataset_name}_val.csv")
    print(f"   ├── {CONFIG.dataset_name}_test.csv")
    print(f"   ├── {CONFIG.dataset_name}_metadata.json")
    print(f"   ├── {CONFIG.dataset_name}_scaler.pkl")
    print(f"   └── preprocessing_report.txt")
    print()
    print("🎯 Ready for ML model training!")
    print()
    print("Next steps:")
    print("  1. Load data: pd.read_csv('data/rocket_flight_ml_train.csv')")
    print("  2. Train models on training set")
    print("  3. Validate on validation set")
    print("  4. Final evaluation on test set")
    print()
    
    return 0


# ============================================================
# QUICK START FUNCTIONS
# ============================================================

def quick_test_pipeline():
    """Quick test with minimal simulations"""
    global SIMULATION_STRATEGY, CONFIG
    SIMULATION_STRATEGY = "quick_test"
    CONFIG.augmentation_factor = 1
    CONFIG.add_sensor_noise = False
    return run_complete_pipeline()


def production_pipeline():
    """Full production pipeline with all features"""
    global SIMULATION_STRATEGY, CONFIG
    SIMULATION_STRATEGY = "comprehensive"
    CONFIG.augmentation_factor = 5
    CONFIG.add_sensor_noise = True
    CONFIG.sensor_noise_level = "medium"
    return run_complete_pipeline()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete ML dataset generation pipeline")
    parser.add_argument('--mode', choices=['normal', 'quick', 'production'], 
                       default='normal', help='Pipeline mode')
    parser.add_argument('--ork', type=str, help='Path to .ork file')
    parser.add_argument('--strategy', type=str, help='Simulation strategy')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Override configuration from command line
    if args.ork:
        ORK_FILE = args.ork
    if args.strategy:
        SIMULATION_STRATEGY = args.strategy
    if args.output:
        CONFIG.output_dir = args.output
    
    # Run pipeline based on mode
    if args.mode == 'quick':
        print("🚀 Running QUICK TEST mode (minimal simulations)\n")
        exit_code = quick_test_pipeline()
    elif args.mode == 'production':
        print("🏭 Running PRODUCTION mode (comprehensive dataset)\n")
        exit_code = production_pipeline()
    else:
        exit_code = run_complete_pipeline()
    
    sys.exit(exit_code)
