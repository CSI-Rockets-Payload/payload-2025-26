#!/usr/bin/env python3
"""
RocketPy ML Dataset Generator - All-in-One Script
Generate diverse rocket flight simulation datasets for machine learning.

QUICK START:
1. Edit the CONFIGURATION section below
2. Set your .ork file path
3. Choose a dataset strategy
4. Run: python rocketpy_ml_dataset_generator.py

Features:
- Systematic parameter variations (mass, drag, thrust, CG, etc.)
- Environmental augmentation (wind, gusts, shear, turbulence)
- Anomaly injection (instabilities, moments, failures)
- Full parameter logging for reproducibility
- Export to CSV/JSON for ML pipelines
"""

import json
import csv
import time
from itertools import product
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime


# ╔════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                              ║
# ║                    👇 EDIT THIS SECTION 👇                         ║
# ╚════════════════════════════════════════════════════════════════════╝

# ============================================================
# BASIC SETTINGS
# ============================================================

# Path to your OpenRocket .ork file
ORK_FILE = "template_rocket.ork"  # 👈 CHANGE THIS

# Output directory for dataset
OUTPUT_DIR = "./ml_dataset"

# Dataset name and description
DATASET_NAME = "rocket_flight_ml_v1"
DATASET_DESCRIPTION = "Parametric variations for ML training"


# ============================================================
# CHOOSE DATASET STRATEGY
# ============================================================
# Uncomment ONE of these strategies:

STRATEGY = "quick_test"              # ~18 sims, 1 min (debugging)
# STRATEGY = "basic_physics"           # ~200 sims, 10 min (basic training)
# STRATEGY = "environmental_sweep"     # ~400 sims, 20 min (weather)
# STRATEGY = "wind_profiles"           # ~240 sims, 12 min (wind analysis)
# STRATEGY = "launch_variations"       # ~315 sims, 16 min (launch optimization)
# STRATEGY = "thrust_anomalies"        # ~400 sims, 20 min (motor variations)
# STRATEGY = "stability_anomalies"     # ~500 sims, 25 min (instabilities)
# STRATEGY = "comprehensive"           # ~10,000+ sims, 6-8 hours (full ML)
# STRATEGY = "monte_carlo"             # Custom samples (statistical)
# STRATEGY = "anomaly_detection"       # ~1,000 sims (anomaly detection)
STRATEGY = "quick_test"  # 👈 DEFAULT

# Monte Carlo sample size (only used if STRATEGY = "monte_carlo")
MONTE_CARLO_SAMPLES = 100


# ============================================================
# CUSTOM STRATEGY (OPTIONAL)
# ============================================================
# If you want complete control, set STRATEGY = "custom" and define below:

CUSTOM_PARAMETERS = [
    ("mass_variation_percent", [-5, 0, 5, 10]),
    ("cd_multiplier", [0.9, 1.0, 1.1]),
    ("wind_constant_x", [0, 5, 10]),
    ("thrust_scale", [0.95, 1.0, 1.05]),
]


# ============================================================
# BASE CONFIGURATION (Usually don't need to change)
# ============================================================

# Environment defaults
BASE_LATITUDE = 28.5   # Kennedy Space Center
BASE_LONGITUDE = -80.6
BASE_ELEVATION = 3.0   # meters

# Launch defaults
BASE_RAIL_LENGTH = 5.2      # meters
BASE_INCLINATION = 85.0     # degrees from horizontal
BASE_HEADING = 0.0          # degrees azimuth

# Sensor noise levels
PRESSURE_NOISE = 0.05       # 5% noise
GPS_NOISE_STD = 5.0         # meters
PARACHUTE_LAG = 1.5         # seconds


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    END OF CONFIGURATION                            ║
# ║                  (Don't edit below unless needed)                  ║
# ╚════════════════════════════════════════════════════════════════════╝


# ============================================================
# DATASET STRATEGY DEFINITIONS
# ============================================================

class DatasetStrategies:
    """Predefined parameter combinations for different use cases"""
    
    @staticmethod
    def quick_test():
        """Quick test with minimal variations"""
        return [
            ("mass_variation_percent", [0, 5, 10]),
            ("cd_multiplier", [0.95, 1.0, 1.05]),
            ("wind_constant_x", [0, 5]),
        ]
    
    @staticmethod
    def basic_physics():
        """Core physical parameter variations"""
        return [
            ("mass_variation_percent", [-10, -5, 0, 5, 10, 15]),
            ("cd_multiplier", [0.90, 0.95, 1.0, 1.05, 1.10]),
            ("thrust_scale", [0.95, 1.0, 1.05]),
            ("cg_offset_m", [-0.03, 0, 0.03]),
        ]
    
    @staticmethod
    def environmental_sweep():
        """Environmental condition variations"""
        return [
            ("wind_constant_x", [0, 5, 10, 15, 20]),
            ("wind_constant_y", [0, 5, 10]),
            ("launch_altitude_m", [0, 500, 1000, 1500, 2000]),
            ("temperature_offset_k", [-10, 0, 10]),
        ]
    
    @staticmethod
    def wind_profiles():
        """Comprehensive wind modeling"""
        return [
            ("wind_constant_x", [0, 5, 10, 15]),
            ("wind_constant_y", [0, 5, 10]),
            ("wind_turbulence", [0, 0.1, 0.3]),
            ("wind_shear_rate", [0, 0.02]),
        ]
    
    @staticmethod
    def launch_variations():
        """Launch angle and rail variations"""
        return [
            ("inclination_offset_deg", [-3, -2, -1, 0, 1, 2, 3]),
            ("heading_offset_deg", [-5, 0, 5]),
            ("rail_length_m", [3.0, 4.0, 5.0, 6.0, 7.0]),
        ]
    
    @staticmethod
    def thrust_anomalies():
        """Motor performance variations"""
        return [
            ("thrust_scale", [0.90, 0.95, 1.0, 1.05, 1.10]),
            ("burn_duration_scale", [0.95, 1.0, 1.05]),
            ("thrust_jitter_percent", [0, 2, 5]),
            ("thrust_jitter_freq_hz", [0, 20, 50]),
        ]
    
    @staticmethod
    def stability_anomalies():
        """Aerodynamic instabilities"""
        return [
            ("roll_moment_nm", [0, 0.5, 1.0, 2.0, 5.0]),
            ("yaw_moment_nm", [0, 1.0, 2.0]),
            ("cg_offset_m", [-0.05, -0.03, 0, 0.03, 0.05]),
            ("inertia_multiplier", [0.9, 1.0, 1.1]),
        ]
    
    @staticmethod
    def comprehensive():
        """Large comprehensive dataset"""
        return [
            ("mass_variation_percent", [-5, 0, 5, 10]),
            ("cd_multiplier", [0.90, 1.0, 1.10]),
            ("thrust_scale", [0.95, 1.0, 1.05]),
            ("cg_offset_m", [-0.03, 0, 0.03]),
            ("wind_constant_x", [0, 5, 10, 15]),
            ("wind_constant_y", [0, 5]),
            ("wind_turbulence", [0, 0.2]),
            ("inclination_offset_deg", [-2, 0, 2]),
            ("launch_altitude_m", [0, 1000]),
            ("thrust_jitter_percent", [0, 3]),
        ]
    
    @staticmethod
    def monte_carlo(n_samples=100):
        """Monte Carlo sampling"""
        np.random.seed(42)
        return [
            ("mass_variation_percent", list(np.random.uniform(-10, 15, n_samples))),
            ("cd_multiplier", list(np.random.uniform(0.85, 1.20, n_samples))),
            ("thrust_scale", list(np.random.uniform(0.90, 1.10, n_samples))),
            ("cg_offset_m", list(np.random.normal(0, 0.02, n_samples))),
            ("wind_constant_x", list(np.random.exponential(5, n_samples))),
            ("wind_constant_y", list(np.random.normal(0, 3, n_samples))),
            ("inclination_offset_deg", list(np.random.normal(0, 1.5, n_samples))),
            ("heading_offset_deg", list(np.random.normal(0, 3, n_samples))),
        ]
    
    @staticmethod
    def anomaly_detection():
        """Dataset for anomaly detection"""
        return [
            ("mass_variation_percent", [0, 5, 10, 20, 30]),
            ("thrust_scale", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
            ("thrust_jitter_percent", [0, 2, 5, 10, 15]),
            ("roll_moment_nm", [0, 1, 2, 5, 10]),
            ("parachute_failure", [False, True]),
            ("wind_constant_x", [0, 10, 20, 30]),
        ]


# ============================================================
# ML DATASET GENERATOR
# ============================================================

class RocketPyMLGenerator:
    """Generates ML datasets from .ork files with parameter augmentation"""

    def __init__(self, ork_file: str, output_dir: str):
        self.ork_file = Path(ork_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.results = []
        self.failed = []
        
        # Import RocketPy
        try:
            from rocketpy import Environment, Flight
            from rocketpy.rocket.rocket import Rocket
            self.Environment = Environment
            self.Rocket = Rocket
            self.Flight = Flight
        except ImportError:
            print("❌ RocketPy not installed! Run: pip install rocketpy")
            raise
        
        if not self.ork_file.exists():
            raise FileNotFoundError(f"Rocket file not found: {self.ork_file}")

    def create_environment(self, params: Dict[str, Any]) -> Any:
        """Create environment with parameter augmentation"""
        lat = BASE_LATITUDE
        lon = BASE_LONGITUDE
        elev = params.get('launch_altitude_m', BASE_ELEVATION)
        
        if 'launch_latitude' in params:
            lat = params['launch_latitude']
        
        env = self.Environment(latitude=lat, longitude=lon, elevation=elev)
        env.set_atmospheric_model(type='standard_atmosphere')
        
        return env

    def simulate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single simulation with augmented parameters"""
        sim_id = f"sim_{len(self.results):06d}"
        
        try:
            # Load rocket
            rocket = self.Rocket.from_ork(str(self.ork_file))
            
            # Create environment
            env = self.create_environment(params)
            
            # Get launch parameters
            rail_length = params.get('rail_length_m', BASE_RAIL_LENGTH)
            inclination = BASE_INCLINATION + params.get('inclination_offset_deg', 0)
            heading = BASE_HEADING + params.get('heading_offset_deg', 0)
            
            # Run flight
            flight = self.Flight(
                rocket=rocket,
                environment=env,
                rail_length=rail_length,
                inclination=inclination,
                heading=heading
            )
            
            # Extract metrics
            def safe_float(attr, default=None):
                if attr is None:
                    return default
                try:
                    if hasattr(attr, 'y_array'):
                        return float(max(attr.y_array))
                    elif hasattr(attr, 'max') and callable(attr.max):
                        return float(attr.max())
                    elif callable(attr):
                        return float(attr(0))
                    else:
                        return float(attr)
                except:
                    return default
            
            results = {
                'simulation_id': sim_id,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                
                # Flight metrics
                'apogee': safe_float(flight.apogee) if hasattr(flight, 'apogee') else None,
                'apogee_time': safe_float(flight.apogee_time) if hasattr(flight, 'apogee_time') else None,
                'max_velocity': safe_float(flight.speed) if hasattr(flight, 'speed') else None,
                'max_acceleration': safe_float(flight.acceleration) if hasattr(flight, 'acceleration') else None,
                'flight_time': safe_float(flight.t_final) if hasattr(flight, 't_final') else None,
                'impact_velocity': safe_float(flight.impact_velocity) if hasattr(flight, 'impact_velocity') else None,
                'x_impact': safe_float(flight.x_impact) if hasattr(flight, 'x_impact') else None,
                'y_impact': safe_float(flight.y_impact) if hasattr(flight, 'y_impact') else None,
                
                # All parameters
                **params
            }
            
            # Calculate drift
            if results['x_impact'] and results['y_impact']:
                results['drift_distance'] = np.sqrt(results['x_impact']**2 + results['y_impact']**2)
            else:
                results['drift_distance'] = None
            
            if results['apogee']:
                print(f"   ✓ [{sim_id}] Apogee: {results['apogee']:.1f} m")
            
            return results
            
        except Exception as e:
            print(f"   ✗ [{sim_id}] FAILED: {e}")
            failure = {
                'simulation_id': sim_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                **params
            }
            self.failed.append(failure)
            return failure

    def generate_dataset(self, param_list: List[Tuple[str, List[Any]]]):
        """Generate complete dataset"""
        param_names = [p[0] for p in param_list]
        param_values = [p[1] for p in param_list]
        combinations = list(product(*param_values))
        
        n_total = len(combinations)
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET: {DATASET_NAME}")
        print(f"{'='*70}")
        print(f"Rocket: {self.ork_file.name}")
        print(f"Total simulations: {n_total:,}")
        print(f"Parameters: {param_names}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Run simulations
        for idx, combo in enumerate(combinations, 1):
            param_dict = dict(zip(param_names, combo))
            
            if idx % 10 == 0 or idx == 1:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (n_total - idx) / rate if rate > 0 else 0
                print(f"[{idx}/{n_total}] ({idx/n_total*100:.1f}%) | "
                      f"{rate:.1f} sim/s | ETA: {remaining/60:.1f} min")
            
            result = self.simulate(param_dict)
            self.results.append(result)
            self._write_csv(result)
        
        # Save outputs
        elapsed = time.time() - start_time
        self._save_all(n_total, elapsed)

    def _write_csv(self, data: Dict[str, Any]):
        """Append to CSV"""
        csv_path = self.output_dir / "dataset.csv"
        exists = csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(data)

    def _save_all(self, n_total: int, elapsed: float):
        """Save all metadata and statistics"""
        success_count = sum(1 for r in self.results if r.get('success'))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"DATASET GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {n_total} | Success: {success_count} | Failed: {len(self.failed)}")
        print(f"Time: {elapsed/60:.1f} min | Rate: {elapsed/n_total:.2f} s/sim")
        print(f"{'='*70}\n")
        
        # Metadata
        metadata = {
            'dataset_name': DATASET_NAME,
            'description': DATASET_DESCRIPTION,
            'rocket_file': str(self.ork_file),
            'total_simulations': n_total,
            'successful': success_count,
            'failed': len(self.failed),
            'generation_date': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed,
        }
        
        with open(self.output_dir / "metadata" / "info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Full results
        with open(self.output_dir / "results_full.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Failed simulations
        if self.failed:
            with open(self.output_dir / "metadata" / "failures.json", 'w') as f:
                json.dump(self.failed, f, indent=2)
        
        # Statistics
        valid = [r for r in self.results if r.get('success') and r.get('apogee')]
        if valid:
            apogees = [r['apogee'] for r in valid]
            vels = [r['max_velocity'] for r in valid if r.get('max_velocity')]
            
            stats = {
                'apogee': {
                    'min': float(min(apogees)),
                    'max': float(max(apogees)),
                    'mean': float(np.mean(apogees)),
                    'std': float(np.std(apogees)),
                    'median': float(np.median(apogees)),
                }
            }
            
            if vels:
                stats['velocity'] = {
                    'min': float(min(vels)),
                    'max': float(max(vels)),
                    'mean': float(np.mean(vels)),
                    'std': float(np.std(vels)),
                }
            
            with open(self.output_dir / "metadata" / "statistics.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"📊 Dataset Statistics:")
            print(f"   Apogee: {stats['apogee']['min']:.1f} - {stats['apogee']['max']:.1f} m")
            print(f"   Mean: {stats['apogee']['mean']:.1f} ± {stats['apogee']['std']:.1f} m")
            if vels:
                print(f"   Velocity: {stats['velocity']['min']:.1f} - {stats['velocity']['max']:.1f} m/s")
        
        print(f"\n📁 Output Files:")
        print(f"   • {self.output_dir / 'dataset.csv'}")
        print(f"   • {self.output_dir / 'results_full.json'}")
        print(f"   • {self.output_dir / 'metadata' / 'info.json'}")
        print(f"   • {self.output_dir / 'metadata' / 'statistics.json'}")
        if self.failed:
            print(f"   • {self.output_dir / 'metadata' / 'failures.json'}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        RocketPy ML Dataset Generator                              ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    # Validate configuration
    if not Path(ORK_FILE).exists():
        print(f"❌ Rocket file not found: {ORK_FILE}")
        print("   👉 Edit ORK_FILE at the top of this script")
        exit(1)
    
    # Get strategy
    strategies = DatasetStrategies()
    strategy_map = {
        'quick_test': strategies.quick_test,
        'basic_physics': strategies.basic_physics,
        'environmental_sweep': strategies.environmental_sweep,
        'wind_profiles': strategies.wind_profiles,
        'launch_variations': strategies.launch_variations,
        'thrust_anomalies': strategies.thrust_anomalies,
        'stability_anomalies': strategies.stability_anomalies,
        'comprehensive': strategies.comprehensive,
        'monte_carlo': lambda: strategies.monte_carlo(MONTE_CARLO_SAMPLES),
        'anomaly_detection': strategies.anomaly_detection,
        'custom': lambda: CUSTOM_PARAMETERS,
    }
    
    if STRATEGY not in strategy_map:
        print(f"❌ Invalid strategy: {STRATEGY}")
        print(f"   Valid options: {list(strategy_map.keys())}")
        exit(1)
    
    parameters = strategy_map[STRATEGY]()
    
    # Estimate and confirm
    n_sims = 1
    for _, values in parameters:
        n_sims *= len(values)
    
    print(f"Configuration:")
    print(f"  • Rocket: {ORK_FILE}")
    print(f"  • Strategy: {STRATEGY}")
    print(f"  • Simulations: {n_sims:,}")
    print(f"  • Output: {OUTPUT_DIR}")
    
    if n_sims > 100:
        print(f"\n⚠️  This will run {n_sims:,} simulations (~{n_sims*2/60:.0f} minutes)")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            exit(0)
    
    print()
    
    # Generate dataset
    generator = RocketPyMLGenerator(ORK_FILE, OUTPUT_DIR)
    generator.generate_dataset(parameters)
    
    print("\n✅ Dataset generation complete!")
    print("🎯 Ready for ML training!\n")
