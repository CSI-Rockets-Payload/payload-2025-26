#!/usr/bin/env python3
"""
Data Preprocessing and Train/Test Split Pipeline
Processes raw simulation data, adds sensor noise, and creates ML-ready datasets.

Features:
- Sensor noise injection (IMU, barometer, GPS)
- Train/validation/test splits with stratification
- Data normalization and feature engineering
- Metadata tracking for reproducibility
- Export to multiple formats (CSV, HDF5, NPY)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import sensor noise module (assumes it's in the same directory)
try:
    from sensor_noise import SensorNoiseSimulator, AccelerometerNoise, GyroscopeNoise
except ImportError:
    print("⚠️  sensor_noise.py not found. Sensor noise will be skipped.")
    SensorNoiseSimulator = None


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing pipeline"""
    # Train/val/test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Sensor noise settings
    add_sensor_noise: bool = True
    sensor_noise_level: str = 'medium'  # 'low', 'medium', 'high'
    imu_sample_rate: float = 100.0  # Hz
    
    # Data augmentation
    augment_training_data: bool = True
    augmentation_factor: int = 3  # How many noisy versions per sample
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Output formats
    save_csv: bool = True
    save_hdf5: bool = True
    save_numpy: bool = False
    
    # Stratification
    stratify_by: Optional[str] = None  # e.g., 'apogee_bin' for balanced splits


# ============================================================
# DATA PREPROCESSOR
# ============================================================

class RocketDataPreprocessor:
    """
    Preprocesses rocket flight simulation data for ML training.
    Handles sensor noise, train/test splits, and normalization.
    """
    
    def __init__(self, config: PreprocessConfig = None):
        """
        Initialize preprocessor
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessConfig()
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Track preprocessing metadata
        self.metadata = {
            'preprocessing_date': datetime.now().isoformat(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'random_seed': self.config.random_seed,
            'train_ratio': self.config.train_ratio,
            'val_ratio': self.config.val_ratio,
            'test_ratio': self.config.test_ratio,
        }
    
    def load_raw_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load raw simulation dataset from CSV"""
        print(f"📂 Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    def add_sensor_noise_to_dataset(self, 
                                    df: pd.DataFrame,
                                    n_augmentations: int = 1) -> pd.DataFrame:
        """
        Add sensor noise to create multiple noisy versions
        
        Args:
            df: Input dataframe
            n_augmentations: Number of noisy versions per sample
            
        Returns:
            Augmented dataframe with sensor noise
        """
        if not self.config.add_sensor_noise or SensorNoiseSimulator is None:
            print("⚠️  Sensor noise disabled or module not available")
            return df
        
        print(f"🔊 Adding sensor noise ({self.config.sensor_noise_level} level)")
        print(f"   Creating {n_augmentations} noisy versions per sample...")
        
        augmented_dfs = [df.copy()]  # Original data
        
        # Noise level configurations
        noise_levels = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }
        noise_multiplier = noise_levels.get(self.config.sensor_noise_level, 1.0)
        
        for aug_idx in range(n_augmentations):
            df_noisy = df.copy()
            
            # Add random noise to relevant columns
            # Accelerometer-like noise
            accel_cols = [c for c in df.columns if 'acceleration' in c.lower() or 'accel' in c.lower()]
            for col in accel_cols:
                if df[col].notna().any():
                    noise_std = df[col].std() * 0.05 * noise_multiplier
                    df_noisy[col] += np.random.normal(0, noise_std, len(df))
            
            # Velocity noise
            vel_cols = [c for c in df.columns if 'velocity' in c.lower() or 'speed' in c.lower()]
            for col in vel_cols:
                if df[col].notna().any():
                    noise_std = df[col].std() * 0.02 * noise_multiplier
                    df_noisy[col] += np.random.normal(0, noise_std, len(df))
            
            # Altitude/position noise (barometer-like)
            alt_cols = [c for c in df.columns if 'altitude' in c.lower() or 'apogee' in c.lower() or 'height' in c.lower()]
            for col in alt_cols:
                if df[col].notna().any():
                    noise_std = 1.0 * noise_multiplier  # meters
                    df_noisy[col] += np.random.normal(0, noise_std, len(df))
            
            # Gyroscope-like noise for angular rates
            gyro_cols = [c for c in df.columns if 'angular' in c.lower() or 'rotation' in c.lower()]
            for col in gyro_cols:
                if df[col].notna().any():
                    noise_std = df[col].std() * 0.03 * noise_multiplier
                    df_noisy[col] += np.random.normal(0, noise_std, len(df))
            
            # Mark this as an augmented sample
            df_noisy['augmentation_id'] = aug_idx + 1
            
            augmented_dfs.append(df_noisy)
        
        # Combine all augmented versions
        result = pd.concat(augmented_dfs, ignore_index=True)
        
        # Add original marker
        result['augmentation_id'] = result.get('augmentation_id', 0).fillna(0)
        
        print(f"   ✓ Augmented from {len(df)} to {len(result)} samples")
        
        return result
    
    def create_stratification_bins(self, 
                                   df: pd.DataFrame,
                                   column: str = 'apogee',
                                   n_bins: int = 5) -> pd.Series:
        """
        Create stratification bins for balanced splits
        
        Args:
            df: Input dataframe
            column: Column to stratify by
            n_bins: Number of bins
            
        Returns:
            Bin assignments for each sample
        """
        if column not in df.columns:
            print(f"⚠️  Column '{column}' not found for stratification")
            return None
        
        # Create quantile-based bins
        bins = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
        return bins
    
    def train_val_test_split(self,
                            df: pd.DataFrame,
                            stratify_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets
        
        Args:
            df: Input dataframe
            stratify_column: Column to stratify by (optional)
            
        Returns:
            (train_df, val_df, test_df)
        """
        print(f"\n📊 Splitting dataset...")
        print(f"   Ratios: Train={self.config.train_ratio:.0%}, "
              f"Val={self.config.val_ratio:.0%}, Test={self.config.test_ratio:.0%}")
        
        # Prepare stratification
        stratify = None
        if stratify_column and stratify_column in df.columns:
            stratify = self.create_stratification_bins(df, stratify_column)
            print(f"   Stratifying by: {stratify_column}")
        
        # First split: train vs (val+test)
        train_size = self.config.train_ratio
        temp_size = self.config.val_ratio + self.config.test_ratio
        
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=self.config.random_seed,
            stratify=stratify,
            shuffle=True
        )
        
        # Second split: val vs test
        val_ratio_adjusted = self.config.val_ratio / temp_size
        
        # Re-stratify for second split if needed
        stratify_temp = None
        if stratify is not None:
            stratify_temp = stratify.loc[temp_df.index]
        
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=self.config.random_seed,
            stratify=stratify_temp,
            shuffle=True
        )
        
        print(f"   ✓ Train: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        print(f"   ✓ Val:   {len(val_df)} samples ({len(val_df)/len(df):.1%})")
        print(f"   ✓ Test:  {len(test_df)} samples ({len(test_df)/len(df):.1%})")
        
        # Update metadata
        self.metadata['n_train'] = len(train_df)
        self.metadata['n_val'] = len(val_df)
        self.metadata['n_test'] = len(test_df)
        self.metadata['n_total'] = len(df)
        
        return train_df, val_df, test_df
    
    def normalize_features(self,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          feature_columns: List[str],
                          target_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using training set statistics
        
        Args:
            train_df, val_df, test_df: Data splits
            feature_columns: Columns to normalize
            target_columns: Target columns (optional, usually don't normalize)
            
        Returns:
            (train_df_norm, val_df_norm, test_df_norm) with normalized features
        """
        if not self.config.normalize_features:
            print("⚠️  Feature normalization disabled")
            return train_df, val_df, test_df
        
        print(f"\n🔧 Normalizing features (method: {self.config.normalization_method})...")
        
        # Initialize scaler
        if self.config.normalization_method == 'standard':
            self.feature_scaler = StandardScaler()
        elif self.config.normalization_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.feature_scaler = MinMaxScaler()
        elif self.config.normalization_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.feature_scaler = RobustScaler()
        else:
            print(f"⚠️  Unknown normalization method: {self.config.normalization_method}")
            return train_df, val_df, test_df
        
        # Fit on training data only
        self.feature_scaler.fit(train_df[feature_columns])
        
        # Transform all splits
        train_norm = train_df.copy()
        val_norm = val_df.copy()
        test_norm = test_df.copy()
        
        train_norm[feature_columns] = self.feature_scaler.transform(train_df[feature_columns])
        val_norm[feature_columns] = self.feature_scaler.transform(val_df[feature_columns])
        test_norm[feature_columns] = self.feature_scaler.transform(test_df[feature_columns])
        
        print(f"   ✓ Normalized {len(feature_columns)} features")
        
        # Store normalization parameters
        self.metadata['normalization_method'] = self.config.normalization_method
        self.metadata['normalized_features'] = feature_columns
        if hasattr(self.feature_scaler, 'mean_'):
            self.metadata['feature_means'] = self.feature_scaler.mean_.tolist()
            self.metadata['feature_stds'] = self.feature_scaler.scale_.tolist()
        
        return train_norm, val_norm, test_norm
    
    def save_datasets(self,
                     train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     output_dir: str,
                     dataset_name: str = "rocket_flight"):
        """
        Save processed datasets in multiple formats
        
        Args:
            train_df, val_df, test_df: Processed data splits
            output_dir: Output directory
            dataset_name: Base name for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving datasets to: {output_path}")
        
        # Save CSV
        if self.config.save_csv:
            train_df.to_csv(output_path / f"{dataset_name}_train.csv", index=False)
            val_df.to_csv(output_path / f"{dataset_name}_val.csv", index=False)
            test_df.to_csv(output_path / f"{dataset_name}_test.csv", index=False)
            print(f"   ✓ Saved CSV files")
        
        # Save HDF5 (efficient for large datasets)
        if self.config.save_hdf5:
            try:
                train_df.to_hdf(output_path / f"{dataset_name}_train.h5", key='data', mode='w')
                val_df.to_hdf(output_path / f"{dataset_name}_val.h5", key='data', mode='w')
                test_df.to_hdf(output_path / f"{dataset_name}_test.h5", key='data', mode='w')
                print(f"   ✓ Saved HDF5 files")
            except Exception as e:
                print(f"   ⚠️  Could not save HDF5: {e}")
        
        # Save NumPy arrays (for deep learning)
        if self.config.save_numpy:
            # Identify feature and target columns
            feature_cols = [c for c in train_df.columns if c not in ['simulation_id', 'timestamp', 'success', 'error']]
            
            np.save(output_path / f"{dataset_name}_train_features.npy", train_df[feature_cols].values)
            np.save(output_path / f"{dataset_name}_val_features.npy", val_df[feature_cols].values)
            np.save(output_path / f"{dataset_name}_test_features.npy", test_df[feature_cols].values)
            print(f"   ✓ Saved NumPy arrays")
        
        # Save metadata
        metadata_path = output_path / f"{dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"   ✓ Saved metadata")
        
        # Save scaler if normalized
        if self.feature_scaler is not None:
            import pickle
            scaler_path = output_path / f"{dataset_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            print(f"   ✓ Saved feature scaler")
        
        print(f"\n✅ All datasets saved successfully!")
    
    def generate_summary_report(self,
                               train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               output_dir: str):
        """Generate comprehensive summary report"""
        output_path = Path(output_dir)
        
        report = []
        report.append("="*70)
        report.append("DATASET PREPROCESSING SUMMARY")
        report.append("="*70)
        report.append(f"Generated: {self.metadata['preprocessing_date']}")
        report.append(f"Random Seed: {self.config.random_seed}")
        report.append("")
        
        report.append("Dataset Splits:")
        report.append(f"  Train:      {len(train_df):6,} samples ({len(train_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
        report.append(f"  Validation: {len(val_df):6,} samples ({len(val_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
        report.append(f"  Test:       {len(test_df):6,} samples ({len(test_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
        report.append(f"  Total:      {len(train_df)+len(val_df)+len(test_df):6,} samples")
        report.append("")
        
        report.append("Preprocessing Configuration:")
        report.append(f"  Sensor noise:     {self.config.add_sensor_noise} (level: {self.config.sensor_noise_level})")
        report.append(f"  Augmentation:     {self.config.augment_training_data} (factor: {self.config.augmentation_factor})")
        report.append(f"  Normalization:    {self.config.normalize_features} (method: {self.config.normalization_method})")
        report.append(f"  Stratification:   {self.config.stratify_by or 'None'}")
        report.append("")
        
        # Feature statistics
        report.append("Feature Statistics (Training Set):")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # Show first 10
            mean = train_df[col].mean()
            std = train_df[col].std()
            report.append(f"  {col:30s}: {mean:10.2f} ± {std:8.2f}")
        if len(numeric_cols) > 10:
            report.append(f"  ... and {len(numeric_cols)-10} more features")
        
        report.append("="*70)
        
        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open(output_path / "preprocessing_report.txt", 'w') as f:
            f.write(report_text)


# ============================================================
# MAIN PIPELINE
# ============================================================

def preprocess_pipeline(input_csv: str,
                       output_dir: str = "./data",
                       config: PreprocessConfig = None):
    """
    Complete preprocessing pipeline
    
    Args:
        input_csv: Path to raw simulation CSV
        output_dir: Output directory for processed data
        config: Preprocessing configuration
    """
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        Rocket Flight Data Preprocessing Pipeline                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    # Initialize preprocessor
    config = config or PreprocessConfig()
    preprocessor = RocketDataPreprocessor(config)
    
    # Load data
    df = preprocessor.load_raw_dataset(input_csv)
    
    # Add sensor noise and augment
    if config.augment_training_data and config.add_sensor_noise:
        df = preprocessor.add_sensor_noise_to_dataset(df, n_augmentations=config.augmentation_factor)
    
    # Create train/val/test splits
    train_df, val_df, test_df = preprocessor.train_val_test_split(
        df,
        stratify_column=config.stratify_by
    )
    
    # Identify feature columns (exclude metadata)
    exclude_cols = {'simulation_id', 'timestamp', 'success', 'error', 'augmentation_id'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Normalize features
    if config.normalize_features:
        train_df, val_df, test_df = preprocessor.normalize_features(
            train_df, val_df, test_df,
            feature_cols
        )
    
    # Save datasets
    preprocessor.save_datasets(train_df, val_df, test_df, output_dir)
    
    # Generate report
    preprocessor.generate_summary_report(train_df, val_df, test_df, output_dir)
    
    print("\n🎯 Preprocessing complete! Ready for ML training.")


if __name__ == "__main__":
    # Example usage
    from dataclasses import dataclass
    
    @dataclass
    class PreprocessConfig:
        train_ratio: float = 0.70
        val_ratio: float = 0.15
        test_ratio: float = 0.15
        random_seed: int = 42
        add_sensor_noise: bool = True
        sensor_noise_level: str = 'medium'
        imu_sample_rate: float = 100.0
        augment_training_data: bool = True
        augmentation_factor: int = 3
        normalize_features: bool = True
        normalization_method: str = 'standard'
        save_csv: bool = True
        save_hdf5: bool = True
        save_numpy: bool = False
        stratify_by: Optional[str] = 'apogee'
    
    print("Preprocessing Pipeline")
    print("=" * 60)
    print("Usage:")
    print("  python preprocess_dataset.py")
    print()
    print("Or import and use:")
    print("  from preprocess_dataset import preprocess_pipeline")
    print("  preprocess_pipeline('raw_data.csv', './data')")
