#!/usr/bin/env python3
"""
Parametric OpenRocket Design Generator with Noise Injection and Simulation
Creates multiple .ork files with varying parameters and runs simulations to collect performance data
"""

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import copy
import numpy as np
import csv
import subprocess
import re
import sys


@dataclass
class RocketParameter:
    """Defines a parameter that can be varied"""
    name: str
    values: List[Any]
    xml_path: str  # XPath to the element in the .ork XML
    attribute: str = None  # If modifying an attribute instead of text
    noise_type: str = None  # 'gaussian', 'uniform', 'percentage', None
    noise_amount: float = 0.0  # Stddev for gaussian, range for uniform, percentage for percentage


@dataclass
class NoiseConfig:
    """Configuration for noise injection"""
    noise_type: str  # 'gaussian', 'uniform', 'percentage'
    amount: float  # Meaning depends on noise_type
    
    def apply(self, value: float, rng: np.random.Generator) -> float:
        """Apply noise to a value"""
        if self.noise_type == 'gaussian':
            # Gaussian noise with stddev = amount
            return value + rng.normal(0, self.amount)
        elif self.noise_type == 'uniform':
            # Uniform noise in range [-amount, +amount]
            return value + rng.uniform(-self.amount, self.amount)
        elif self.noise_type == 'percentage':
            # Percentage-based noise (e.g., 0.05 = ±5%)
            noise_range = value * self.amount
            return value + rng.uniform(-noise_range, noise_range)
        else:
            return value


class OpenRocketSimulator:
    """
    Runs OpenRocket simulations and extracts results
    """
    
    def __init__(self, openrocket_jar: str = None):
        """
        Initialize the simulator
        
        Args:
            openrocket_jar: Path to OpenRocket JAR file. If None, searches common locations.
        """
        self.openrocket_jar = openrocket_jar or self._find_openrocket()
        
        if not self.openrocket_jar or not Path(self.openrocket_jar).exists():
            print("⚠️  Warning: OpenRocket JAR not found. Simulations will be skipped.")
            print("   Set openrocket_jar path or place OpenRocket.jar in current directory")
            self.openrocket_jar = None
    
    def _find_openrocket(self) -> Optional[str]:
        """Try to find OpenRocket JAR in common locations"""
        search_paths = [
            "OpenRocket.jar",
            "OpenRocket-15.03.jar",
            "./openrocket/OpenRocket.jar",
            "/Applications/OpenRocket.app/Contents/Resources/OpenRocket.jar",
            str(Path.home() / "OpenRocket" / "OpenRocket.jar")
        ]
        
        for path in search_paths:
            if Path(path).exists():
                return str(Path(path).absolute())
        
        return None
    
    
    def simulate(self, ork_file: Path) -> Optional[Dict[str, Any]]:
        """
        Run simulation on a .ork file and extract results
        
        Args:
            ork_file: Path to .ork file
        
        Returns:
            Dictionary with simulation results, or None if simulation failed
        """
        if not self.openrocket_jar:
            print("   ⚠️  No OpenRocket JAR path specified.")
            return None

        try:
            # Define the output CSV file
            output_csv = ork_file.parent / f"{ork_file.stem}_sim.csv"

            # Build the simulation command
            cmd = [
                "java",
                "--enable-native-access=ALL-UNNAMED",  # Avoids restricted method warnings
                "-jar", str(self.openrocket_jar),
                "--simulate", str(ork_file),
                "--output", str(output_csv)
            ]

            print(f"   🛰️  Running simulation: {ork_file.name}")

            # Run the subprocess with extended timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # Increase timeout from 30 → 120 seconds
                check=False
            )

            # Check for non-zero return code
            if result.returncode != 0:
                print(f"   ⚠️  Simulation failed for {ork_file.name}")
                if result.stderr.strip():
                    print(f"      STDERR: {result.stderr.strip()}")
                return None

            # Parse the CSV output
            if output_csv.exists():
                sim_data = self._parse_simulation_csv(output_csv)
                output_csv.unlink(missing_ok=True)  # Clean up safely
                return sim_data
            else:
                print(f"   ⚠️  No output generated for {ork_file.name}")
                if result.stdout.strip():
                    print(f"      STDOUT: {result.stdout.strip()}")
                return None

        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Simulation timeout for {ork_file.name}")
            return None
        except Exception as e:
            print(f"   ❌  Error simulating {ork_file.name}: {e}")
            return None

    def _parse_simulation_csv(self, csv_file: Path) -> Dict[str, Any]:
        """Parse OpenRocket CSV output and extract key metrics"""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return {}
            
            # Extract key metrics from the simulation data
            times = [float(row.get('Time (s)', 0)) for row in rows]
            altitudes = [float(row.get('Altitude (m)', 0)) for row in rows]
            velocities = [float(row.get('Vertical velocity (m/s)', 0)) for row in rows]
            
            # Find apogee (max altitude)
            max_altitude_idx = altitudes.index(max(altitudes))
            
            results = {
                'max_altitude_m': max(altitudes),
                'apogee_time_s': times[max_altitude_idx],
                'max_velocity_ms': max(velocities),
                'flight_time_s': max(times),
                'landing_velocity_ms': abs(velocities[-1]) if velocities else 0,
            }
            
            # Try to get stability margin from the first row
            if 'Stability margin calibers' in rows[0]:
                results['stability_margin'] = float(rows[0].get('Stability margin calibers', 0))
            
            return results
            
        except Exception as e:
            print(f"   ⚠️  Error parsing CSV: {e}")
            return {}


class ParametricRocketGenerator:
    """
    Generates parametric variations of OpenRocket designs with optional noise
    and runs simulations to collect performance data
    
    OpenRocket .ork files are ZIP archives containing:
    - rocket.ork (XML file with the design)
    - simulations.ork (XML file with simulation settings)
    """
    
    def __init__(self, template_ork: str, noise_seed: Optional[int] = None, 
                 openrocket_jar: str = None):
        """
        Initialize with a template .ork file
        
        Args:
            template_ork: Path to template OpenRocket file
            noise_seed: Random seed for reproducible noise generation
            openrocket_jar: Path to OpenRocket JAR file for simulations
        """
        self.template_path = Path(template_ork)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_ork}")
        
        # Random number generator for noise
        self.rng = np.random.default_rng(noise_seed)
        
        # Initialize simulator
        self.simulator = OpenRocketSimulator(openrocket_jar)
        
        # Load template files
        self.rocket_xml = None
        self.simulations_xml = None
        self._load_template()
    
    def _load_template(self):
        """Load XML from the template .ork file"""
        with zipfile.ZipFile(self.template_path, 'r') as zf:
            # Read rocket design
            with zf.open('rocket.ork') as f:
                self.rocket_xml = ET.parse(f)
            
            # Read simulations (if exists)
            try:
                with zf.open('simulations.ork') as f:
                    self.simulations_xml = ET.parse(f)
            except KeyError:
                self.simulations_xml = None
    
    def modify_parameter(self, tree: ET.ElementTree, xpath: str, 
                        value: Any, attribute: str = None,
                        noise_config: Optional[NoiseConfig] = None):
        """
        Modify a parameter in the XML tree with optional noise
        
        Args:
            tree: XML ElementTree
            xpath: XPath to element (simplified - uses find)
            value: New value to set
            attribute: If set, modify attribute instead of text
            noise_config: Optional noise configuration
        
        Returns:
            Tuple of (success: bool, final_value: Any)
        """
        root = tree.getroot()
        
        # OpenRocket uses nested structure, need to navigate carefully
        element = root.find(xpath)
        
        if element is None:
            # Try more aggressive search
            element = root.find(f".//{xpath.split('/')[-1]}")
        
        if element is not None:
            # Apply noise if configured and value is numeric
            final_value = value
            if noise_config is not None and isinstance(value, (int, float)):
                final_value = noise_config.apply(float(value), self.rng)
            
            if attribute:
                element.set(attribute, str(final_value))
            else:
                element.text = str(final_value)
            return True, final_value
        else:
            print(f"⚠️  Warning: Could not find element at {xpath}")
            return False, value
    
    def _save_metadata_csv(self, metadata: List[Dict], output_path: Path):
        """Save metadata as CSV file"""
        if not metadata:
            return
        
        # Flatten nested dictionaries for CSV
        flattened_rows = []
        for entry in metadata:
            row = {
                'file': entry['file'],
                'index': entry['index']
            }
            
            # Add combo_index and mc_sample if present
            if 'combo_index' in entry:
                row['combo_index'] = entry['combo_index']
            if 'mc_sample' in entry:
                row['mc_sample'] = entry['mc_sample']
            
            # Add nominal parameters
            if 'nominal_parameters' in entry:
                for key, val in entry['nominal_parameters'].items():
                    row[f'nominal_{key}'] = val
            elif 'parameters' in entry:
                for key, val in entry['parameters'].items():
                    row[key] = val
            
            # Add actual parameters if present
            if 'actual_parameters' in entry:
                for key, val in entry['actual_parameters'].items():
                    row[f'actual_{key}'] = val
            
            # Add base_design if present (for noisy variants)
            if 'base_design' in entry:
                for key, val in entry['base_design'].items():
                    row[f'base_{key}'] = val
            
            # Add simulation results if present
            if 'simulation_results' in entry and entry['simulation_results']:
                for key, val in entry['simulation_results'].items():
                    row[key] = val
            
            flattened_rows.append(row)
        
        # Write CSV
        if flattened_rows:
            fieldnames = list(flattened_rows[0].keys())
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_rows)
    
    def generate_design_space(self, parameters: List[RocketParameter], 
                             output_dir: str = "./variants",
                             naming_scheme: str = "rocket_{idx:03d}",
                             global_noise: Optional[NoiseConfig] = None,
                             monte_carlo_samples: int = 1,
                             export_csv: bool = True,
                             run_simulations: bool = True) -> List[Path]:
        """
        Generate all combinations of parameters with optional noise (full factorial design)
        
        Args:
            parameters: List of RocketParameter objects
            output_dir: Directory to save variants
            naming_scheme: How to name files (use {idx} for index)
            global_noise: Apply this noise to all parameters (overridden by parameter-specific noise)
            monte_carlo_samples: Generate N noisy versions of each design point
            export_csv: Also export metadata as CSV file
            run_simulations: Run OpenRocket simulations and collect performance data
        
        Returns:
            List of paths to generated .ork files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all combinations
        param_names = [p.name for p in parameters]
        param_values = [p.values for p in parameters]
        combinations = list(product(*param_values))
        
        total_variants = len(combinations) * monte_carlo_samples
        
        print(f"🚀 Generating {total_variants} design variants...")
        if monte_carlo_samples > 1:
            print(f"   Base combinations: {len(combinations)}")
            print(f"   Monte Carlo samples per combination: {monte_carlo_samples}")
        print(f"   Parameters: {param_names}")
        print(f"   Output: {output_dir}")
        if run_simulations:
            print(f"   Running simulations: {'✓' if self.simulator.openrocket_jar else '✗ (JAR not found)'}")
        
        generated_files = []
        metadata = []
        
        variant_idx = 0
        for combo_idx, combo in enumerate(combinations):
            param_dict = dict(zip(param_names, combo))
            
            # Generate monte carlo samples for this combination
            for mc_sample in range(monte_carlo_samples):
                # Create filename
                if monte_carlo_samples > 1:
                    filename = naming_scheme.format(
                        idx=variant_idx, 
                        combo=combo_idx,
                        sample=mc_sample,
                        **param_dict
                    )
                else:
                    filename = naming_scheme.format(idx=variant_idx, **param_dict)
                
                if not filename.endswith('.ork'):
                    filename += '.ork'
                
                output_path = output_dir / filename
                
                # Generate variant
                rocket_tree = copy.deepcopy(self.rocket_xml)
                
                # Track actual values after noise
                actual_values = {}
                
                # Apply each parameter
                for param, value in zip(parameters, combo):
                    # Determine noise config
                    noise_cfg = None
                    if param.noise_type is not None:
                        noise_cfg = NoiseConfig(param.noise_type, param.noise_amount)
                    elif global_noise is not None and monte_carlo_samples > 1:
                        noise_cfg = global_noise
                    
                    success, final_value = self.modify_parameter(
                        rocket_tree, 
                        param.xml_path, 
                        value, 
                        param.attribute,
                        noise_cfg
                    )
                    
                    actual_values[param.name] = final_value
                
                # Save to .ork file
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    rocket_str = ET.tostring(rocket_tree.getroot(), encoding='unicode')
                    zf.writestr('rocket.ork', rocket_str)
                    
                    if self.simulations_xml:
                        sim_tree = copy.deepcopy(self.simulations_xml)
                        sim_str = ET.tostring(sim_tree.getroot(), encoding='unicode')
                        zf.writestr('simulations.ork', sim_str)
                
                generated_files.append(output_path)
                
                # Run simulation if requested
                sim_results = None
                if run_simulations and self.simulator.openrocket_jar:
                    sim_results = self.simulator.simulate(output_path)
                
                # Store metadata
                meta_entry = {
                    'file': str(output_path),
                    'index': variant_idx,
                    'nominal_parameters': param_dict,
                    'simulation_results': sim_results
                }
                
                if monte_carlo_samples > 1:
                    meta_entry['combo_index'] = combo_idx
                    meta_entry['mc_sample'] = mc_sample
                    meta_entry['actual_parameters'] = actual_values
                else:
                    meta_entry['parameters'] = param_dict
                
                metadata.append(meta_entry)
                
                variant_idx += 1
                
                if variant_idx % 10 == 0 or variant_idx == total_variants:
                    print(f"   Progress: {variant_idx}/{total_variants}")
        
        # Save metadata as JSON
        metadata_path = output_dir / "design_space.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Generated {len(generated_files)} variants")
        print(f"📊 Metadata saved to: {metadata_path}")
        
        # Save metadata as CSV
        if export_csv:
            csv_path = output_dir / "design_space.csv"
            self._save_metadata_csv(metadata, csv_path)
            print(f"📊 CSV results saved to: {csv_path}")
            
            if run_simulations:
                successful_sims = sum(1 for m in metadata if m.get('simulation_results'))
                print(f"🎯 Simulations completed: {successful_sims}/{len(metadata)}")
        
        return generated_files
    
    def generate_noisy_variants(self, base_design: Dict[str, Any],
                               parameters: List[RocketParameter],
                               n_samples: int,
                               output_dir: str = "./noisy_variants",
                               noise_config: Optional[NoiseConfig] = None,
                               export_csv: bool = True,
                               run_simulations: bool = True) -> List[Path]:
        """
        Generate multiple noisy variants around a single design point
        
        Args:
            base_design: Dict of parameter names to base values
            parameters: List of RocketParameter objects (for xml_path info)
            n_samples: Number of noisy variants to generate
            output_dir: Output directory
            noise_config: Noise configuration to apply
            export_csv: Also export metadata as CSV file
            run_simulations: Run OpenRocket simulations and collect performance data
        
        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎲 Generating {n_samples} noisy variants around base design...")
        print(f"   Base design: {base_design}")
        print(f"   Output: {output_dir}")
        if run_simulations:
            print(f"   Running simulations: {'✓' if self.simulator.openrocket_jar else '✗ (JAR not found)'}")
        
        generated_files = []
        metadata = []
        
        for idx in range(n_samples):
            filename = f"noisy_{idx:03d}.ork"
            output_path = output_dir / filename
            
            rocket_tree = copy.deepcopy(self.rocket_xml)
            actual_values = {}
            
            # Apply each parameter with noise
            for param in parameters:
                base_value = base_design.get(param.name)
                if base_value is None:
                    continue
                
                # Use parameter-specific noise or global noise
                noise_cfg = None
                if param.noise_type is not None:
                    noise_cfg = NoiseConfig(param.noise_type, param.noise_amount)
                elif noise_config is not None:
                    noise_cfg = noise_config
                
                success, final_value = self.modify_parameter(
                    rocket_tree,
                    param.xml_path,
                    base_value,
                    param.attribute,
                    noise_cfg
                )
                
                actual_values[param.name] = final_value
            
            # Save to .ork file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                rocket_str = ET.tostring(rocket_tree.getroot(), encoding='unicode')
                zf.writestr('rocket.ork', rocket_str)
                
                if self.simulations_xml:
                    sim_tree = copy.deepcopy(self.simulations_xml)
                    sim_str = ET.tostring(sim_tree.getroot(), encoding='unicode')
                    zf.writestr('simulations.ork', sim_str)
            
            generated_files.append(output_path)
            
            # Run simulation if requested
            sim_results = None
            if run_simulations and self.simulator.openrocket_jar:
                sim_results = self.simulator.simulate(output_path)
            
            metadata.append({
                'file': str(output_path),
                'index': idx,
                'base_design': base_design,
                'actual_parameters': actual_values,
                'simulation_results': sim_results
            })
            
            if (idx + 1) % 10 == 0 or idx == n_samples:
                print(f"   Progress: {idx + 1}/{n_samples}")
        
        # Save metadata as JSON
        metadata_path = output_dir / "noisy_variants.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Generated {len(generated_files)} noisy variants")
        print(f"📊 Metadata saved to: {metadata_path}")
        
        # Save metadata as CSV
        if export_csv:
            csv_path = output_dir / "noisy_variants.csv"
            self._save_metadata_csv(metadata, csv_path)
            print(f"📊 CSV results saved to: {csv_path}")
            
            if run_simulations:
                successful_sims = sum(1 for m in metadata if m.get('simulation_results'))
                print(f"🎯 Simulations completed: {successful_sims}/{len(metadata)}")
        
        return generated_files


def example_parametric_study():
    """Example: Generate variants exploring fin design space"""
    
    template_file = "template_rocket.ork"
    
    # Check if template exists
    if not Path(template_file).exists():
        print(f"❌ Template file not found: {template_file}")
        print("\n📝 Create a template rocket in OpenRocket first, then:")
        print("   1. Save it as 'template_rocket.ork'")
        print("   2. Run this script to generate variants")
        return
    
    # Define parameters to explore
    parameters = [
        RocketParameter(
            name="fin_height",
            values=[0.08, 0.10, 0.12, 0.14],  # meters
            xml_path=".//finset/height"
        ),
        RocketParameter(
            name="fin_count",
            values=[3, 4],
            xml_path=".//finset/fincount"
        ),
        RocketParameter(
            name="body_length",
            values=[0.5, 0.6, 0.7],  # meters
            xml_path=".//bodytube/length"
        )
    ]
    
    # Generate design space
    generator = ParametricRocketGenerator(template_file)
    
    # This creates 4 × 2 × 3 = 24 variants
    variants = generator.generate_design_space(
        parameters,
        output_dir="./parametric_study",
        naming_scheme="rocket_h{fin_height}_n{fin_count}_l{body_length}",
        export_csv=True,
        run_simulations=True  # Run simulations and collect data
    )
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Check parametric_study/design_space.csv for results")
    print(f"   2. Analyze to find optimal design")
    
    return variants


def example_monte_carlo_study():
    """Example: Monte Carlo analysis with manufacturing tolerances"""
    
    template_file = "template_rocket.ork"
    
    if not Path(template_file).exists():
        print(f"❌ Template file not found: {template_file}")
        return
    
    # Define nominal parameters with manufacturing tolerances
    parameters = [
        RocketParameter(
            name="fin_height",
            values=[0.10],  # Single nominal value
            xml_path=".//finset/height",
            noise_type='percentage',
            noise_amount=0.02  # ±2% manufacturing tolerance
        ),
        RocketParameter(
            name="fin_root_chord",
            values=[0.15],
            xml_path=".//finset/rootchord",
            noise_type='percentage',
            noise_amount=0.02
        ),
        RocketParameter(
            name="body_radius",
            values=[0.030],
            xml_path=".//bodytube/radius",
            noise_type='gaussian',
            noise_amount=0.0005  # 0.5mm standard deviation
        )
    ]
    
    generator = ParametricRocketGenerator(template_file, noise_seed=42)
    
    # Generate 100 Monte Carlo samples with manufacturing noise
    variants = generator.generate_design_space(
        parameters,
        output_dir="./monte_carlo_study",
        naming_scheme="rocket_mc_{sample:03d}",
        monte_carlo_samples=100,
        export_csv=True,
        run_simulations=True  # Simulate all variants
    )
    
    print(f"\n🎯 Monte Carlo study complete!")
    print(f"   Check monte_carlo_study/design_space.csv for performance statistics")
    
    return variants


def example_parametric_with_noise():
    """Example: Parametric study with robustness samples at each point"""
    
    template_file = "template_rocket.ork"
    
    if not Path(template_file).exists():
        print(f"❌ Template file not found: {template_file}")
        return
    
    parameters = [
        RocketParameter(
            name="fin_height",
            values=[0.08, 0.10, 0.12],
            xml_path=".//finset/height"
        ),
        RocketParameter(
            name="fin_count",
            values=[3, 4],
            xml_path=".//finset/fincount"
        )
    ]
    
    generator = ParametricRocketGenerator(template_file, noise_seed=42)
    
    # Global noise for robustness: 1% variation on all parameters
    global_noise = NoiseConfig(noise_type='percentage', amount=0.01)
    
    # Generate 6 design points × 10 samples each = 60 variants
    variants = generator.generate_design_space(
        parameters,
        output_dir="./parametric_robust_study",
        naming_scheme="rocket_c{combo:02d}_s{sample:02d}",
        global_noise=global_noise,
        monte_carlo_samples=10,
        export_csv=True,
        run_simulations=True
    )
    
    print(f"\n🎯 Parametric robustness study complete!")
    print(f"   Each design point tested with 10 noisy variants")
    
    return variants


def example_targeted_noise():
    """Example: Add noise around a specific optimal design"""
    
    template_file = "template_rocket.ork"
    
    if not Path(template_file).exists():
        print(f"❌ Template file not found: {template_file}")
        return
    
    # Define the optimal design found from previous studies
    optimal_design = {
        'fin_height': 0.105,
        'fin_root_chord': 0.148,
        'body_radius': 0.0315
    }
    
    parameters = [
        RocketParameter(
            name="fin_height",
            values=[],  # Not used in this mode
            xml_path=".//finset/height"
        ),
        RocketParameter(
            name="fin_root_chord",
            values=[],
            xml_path=".//finset/rootchord"
        ),
        RocketParameter(
            name="body_radius",
            values=[],
            xml_path=".//bodytube/radius"
        )
    ]
    
    generator = ParametricRocketGenerator(template_file, noise_seed=42)
    
    # Generate 50 noisy variants around optimal design
    noise_config = NoiseConfig(noise_type='percentage', amount=0.03)  # ±3%
    
    variants = generator.generate_noisy_variants(
        base_design=optimal_design,
        parameters=parameters,
        n_samples=50,
        output_dir="./optimal_robustness",
        noise_config=noise_config,
        export_csv=True,
        run_simulations=True
    )
    
    print(f"\n🎯 Robustness analysis around optimal design complete!")
    print(f"   Test how stable the optimal performance is to parameter variations")
    
    return variants


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Parametric OpenRocket Design Generator              ║")
    print("║   with Noise Injection & Simulation                   ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("This script generates rocket variants and runs simulations")
    print("to collect performance data for design optimization.")
    print()
    print("NOISE TYPES:")
    print("  • gaussian: Normal distribution (stddev specified)")
    print("  • uniform: Uniform distribution (±range specified)")
    print("  • percentage: Relative noise (e.g., 0.05 = ±5%)")
    print()
    print("COLLECTED METRICS:")
    print("  • Max altitude, apogee time, max velocity")
    print("  • Flight time, landing velocity, stability margin")
    print()
    print("SETUP:")
    print("  1. Create template_rocket.ork in OpenRocket")
    print("  2. Place OpenRocket.jar in this directory (or set path)")
    print("  3. Uncomment an example below and run")
    print()
    
    # Uncomment the example you want to run:
    example_monte_carlo_study()           # Manufacturing tolerance analysis
    # example_parametric_study()          # Basic parametric sweep
    # example_parametric_with_noise()     # Parametric + robustness
    # example_targeted_noise()            # Test optimal design robustness
    
    print("\n💡 Check the CSV file for complete results with simulation data!")
