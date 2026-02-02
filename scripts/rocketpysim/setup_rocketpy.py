#!/usr/bin/env python3
"""
Cross-Platform RocketPy and RocketSerializer Installer
Automatically installs all dependencies needed to run OpenRocket simulations via Python
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import tarfile
from pathlib import Path

class RocketPyInstaller:
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.install_dir = Path.home() / ".rocketpy_deps"
        self.java_version = "17"
        self.openrocket_version = "23.09"
        
    def print_step(self, message):
        """Print a formatted step message"""
        print(f"\n{'='*60}")
        print(f"  {message}")
        print(f"{'='*60}\n")
    
    def check_python_version(self):
        """Ensure Python 3.7+ is installed"""
        self.print_step("Checking Python version")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            print("❌ Python 3.7 or higher is required")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            sys.exit(1)
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    
    def check_java(self):
        """Check if Java 17+ is installed"""
        self.print_step("Checking Java installation")
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True
            )
            version_output = result.stderr
            
            # Extract version number
            if "version" in version_output:
                # Parse version from output like 'java version "17.0.1"'
                import re
                match = re.search(r'version "(\d+)', version_output)
                if match:
                    major_version = int(match.group(1))
                    if major_version >= 17:
                        print(f"✅ Java {major_version} detected")
                        return True
                    else:
                        print(f"⚠️  Java {major_version} found, but version 17+ recommended")
                        return False
        except FileNotFoundError:
            print("❌ Java not found")
            return False
        
        return False
    
    def install_java(self):
        """Provide instructions for Java installation"""
        self.print_step("Java Installation Required")
        
        java_urls = {
            "Windows": "https://download.oracle.com/java/17/latest/jdk-17_windows-x64_bin.exe",
            "Darwin": "https://download.oracle.com/java/17/latest/jdk-17_macos-x64_bin.dmg",
            "Linux": "https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.tar.gz"
        }
        
        print("Java 17 is required but not found on your system.")
        print("\nOption 1 - Automatic download (Oracle JDK):")
        print(f"  {java_urls.get(self.system, 'Not available for your OS')}")
        
        print("\nOption 2 - Package manager (Recommended):")
        if self.system == "Windows":
            print("  Using Chocolatey: choco install openjdk17")
            print("  Using Scoop: scoop install openjdk17")
        elif self.system == "Darwin":
            print("  Using Homebrew: brew install openjdk@17")
        elif self.system == "Linux":
            print("  Ubuntu/Debian: sudo apt install openjdk-17-jdk")
            print("  Fedora: sudo dnf install java-17-openjdk")
            print("  Arch: sudo pacman -S jdk17-openjdk")
        
        print("\nOption 3 - Manual download:")
        print("  https://www.oracle.com/java/technologies/downloads/#java17")
        
        response = input("\nWould you like to open the download page? (y/n): ")
        if response.lower() == 'y':
            import webbrowser
            webbrowser.open("https://www.oracle.com/java/technologies/downloads/#java17")
        
        print("\n⚠️  Please install Java and run this script again.")
        sys.exit(0)
    
    def download_openrocket_jar(self):
        """Download OpenRocket JAR file"""
        self.print_step("Downloading OpenRocket JAR")
        
        # Create installation directory
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        jar_filename = f"OpenRocket-{self.openrocket_version}.jar"
        jar_path = self.install_dir / jar_filename
        
        if jar_path.exists():
            print(f"✅ OpenRocket JAR already exists at {jar_path}")
            return jar_path
        
        # OpenRocket download URL
        jar_url = f"https://github.com/openrocket/openrocket/releases/download/release-{self.openrocket_version}/OpenRocket-{self.openrocket_version}.jar"
        
        print(f"Downloading from: {jar_url}")
        print(f"Saving to: {jar_path}")
        
        try:
            with urllib.request.urlopen(jar_url) as response:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(jar_path, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        f.write(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
                
                print("\n✅ Download complete!")
                return jar_path
                
        except Exception as e:
            print(f"\n❌ Failed to download: {e}")
            print(f"\nPlease manually download from:")
            print(f"https://openrocket.info/downloads.html?vers={self.openrocket_version}#content-JAR")
            print(f"And save it to: {jar_path}")
            sys.exit(1)
    
    def create_venv(self):
        """Create a virtual environment for RocketPy"""
        self.print_step("Setting up Python virtual environment")
        
        venv_path = self.install_dir / "venv"
        pip_path = self.get_venv_pip(venv_path)
        
        # Check if venv exists and is valid
        if venv_path.exists():
            if pip_path.exists():
                print(f"✅ Virtual environment already exists at {venv_path}")
                return venv_path
            else:
                print(f"⚠️  Existing venv is incomplete (no pip), recreating...")
                import shutil
                shutil.rmtree(venv_path)
        
        print(f"Creating virtual environment at {venv_path}...")
        try:
            import venv
            venv.create(venv_path, with_pip=True)
            
            # Verify pip was installed
            if not pip_path.exists():
                raise Exception("pip was not installed in the virtual environment")
            
            print(f"✅ Virtual environment created successfully")
            return venv_path
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            print("\nTry installing python3-venv:")
            if self.system == "Linux":
                print(f"  sudo apt install python3.{sys.version_info.minor}-venv python3-full")
            print("\nAfter installing, delete the incomplete venv and run again:")
            print(f"  rm -rf {venv_path}")
            sys.exit(1)
    
    def get_venv_python(self, venv_path):
        """Get the path to the Python executable in the venv"""
        if self.system == "Windows":
            return venv_path / "Scripts" / "python.exe"
        else:
            return venv_path / "bin" / "python"
    
    def get_venv_pip(self, venv_path):
        """Get the path to pip in the venv"""
        if self.system == "Windows":
            return venv_path / "Scripts" / "pip.exe"
        else:
            return venv_path / "bin" / "pip"
    
    def install_python_packages(self, venv_path):
        """Install Python packages using pip in virtual environment"""
        self.print_step("Installing Python packages")
        
        pip_path = self.get_venv_pip(venv_path)
        packages = ["rocketserializer"]
        
        # Upgrade pip first
        print("Upgrading pip...")
        try:
            subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        except subprocess.CalledProcessError:
            print("⚠️  Could not upgrade pip, continuing anyway...")
        
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([
                    str(pip_path), "install", "--upgrade", package
                ])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                sys.exit(1)
    
    def create_config_file(self, jar_path, venv_path):
        """Create a configuration file with paths"""
        self.print_step("Creating configuration file")
        
        config_path = self.install_dir / "config.txt"
        
        with open(config_path, 'w') as f:
            f.write(f"OPENROCKET_JAR={jar_path}\n")
            f.write(f"INSTALL_DIR={self.install_dir}\n")
            f.write(f"VENV_PATH={venv_path}\n")
            f.write(f"JAVA_VERSION={self.java_version}\n")
            f.write(f"OPENROCKET_VERSION={self.openrocket_version}\n")
        
        print(f"✅ Configuration saved to {config_path}")
    
    def create_example_script(self, venv_path):
        """Create an example usage script"""
        self.print_step("Creating example script")
        
        example_script = self.install_dir / "example_usage.py"
        venv_python = self.get_venv_python(venv_path)
        
        script_content = f'''#!/usr/bin/env python3
"""
Example script for using RocketSerializer with OpenRocket
"""

import subprocess
from pathlib import Path

# Path to OpenRocket JAR (automatically configured)
OPENROCKET_JAR = Path("{self.install_dir}") / "OpenRocket-{self.openrocket_version}.jar"
VENV_PYTHON = Path("{venv_python}")

def convert_ork_to_json(ork_file, output_dir=None):
    """
    Convert an OpenRocket .ork file to JSON format
    
    Args:
        ork_file: Path to your .ork file
        output_dir: Optional output directory
    """
    cmd = [
        str(VENV_PYTHON), "-m", "rocketserializer.cli.ork2json",
        "--filepath", str(ork_file),
        "--ork_jar", str(OPENROCKET_JAR),
        "--verbose", "True"
    ]
    
    if output_dir:
        cmd.extend(["--output", str(output_dir)])
    
    subprocess.run(cmd)

def convert_ork_to_notebook(ork_file, output_dir=None):
    """
    Convert an OpenRocket .ork file to a Jupyter notebook
    
    Args:
        ork_file: Path to your .ork file
        output_dir: Optional output directory
    """
    cmd = [
        str(VENV_PYTHON), "-m", "rocketserializer.cli.ork2notebook",
        "--filepath", str(ork_file),
        "--ork_jar", str(OPENROCKET_JAR),
        "--verbose", "True"
    ]
    
    if output_dir:
        cmd.extend(["--output", str(output_dir)])
    
    subprocess.run(cmd)

if __name__ == "__main__":
    # Example usage
    print("RocketPy + RocketSerializer Example")
    print("=" * 50)
    print(f"OpenRocket JAR: {{OPENROCKET_JAR}}")
    print(f"Python venv: {{VENV_PYTHON}}")
    print()
    print("To convert your .ork file:")
    print('  python example_usage.py')
    print()
    print("Then edit this script to add your rocket file path:")
    print('  ork_file = "path/to/your_rocket.ork"')
    print('  convert_ork_to_json(ork_file)')
    print()
    print("Or create a notebook:")
    print('  convert_ork_to_notebook(ork_file)')
    
    # Uncomment and modify this to use:
    # ork_file = "your_rocket.ork"
    # convert_ork_to_json(ork_file)
'''
        
        with open(example_script, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix-like systems
        if self.system != "Windows":
            os.chmod(example_script, 0o755)
        
        print(f"✅ Example script created at {example_script}")
    
    def create_activation_scripts(self, venv_path):
        """Create convenience scripts to activate the environment"""
        self.print_step("Creating activation helper scripts")
        
        # Create activation script for Unix-like systems
        if self.system != "Windows":
            activate_script = self.install_dir / "activate_rocketpy.sh"
            with open(activate_script, 'w') as f:
                f.write(f'''#!/bin/bash
# Activate RocketPy virtual environment
source {venv_path}/bin/activate
echo "✅ RocketPy environment activated!"
echo "OpenRocket JAR: {self.install_dir}/OpenRocket-{self.openrocket_version}.jar"
echo ""
echo "Usage:"
echo "  ork2json --filepath your_rocket.ork --ork_jar {self.install_dir}/OpenRocket-{self.openrocket_version}.jar"
echo "  ork2notebook --filepath your_rocket.ork --ork_jar {self.install_dir}/OpenRocket-{self.openrocket_version}.jar"
''')
            os.chmod(activate_script, 0o755)
            print(f"✅ Created activation script: {activate_script}")
        
        # Create activation script for Windows
        else:
            activate_script = self.install_dir / "activate_rocketpy.bat"
            with open(activate_script, 'w') as f:
                f.write(f'''@echo off
REM Activate RocketPy virtual environment
call {venv_path}\\Scripts\\activate.bat
echo ✅ RocketPy environment activated!
echo OpenRocket JAR: {self.install_dir}\\OpenRocket-{self.openrocket_version}.jar
echo.
echo Usage:
echo   ork2json --filepath your_rocket.ork --ork_jar {self.install_dir}\\OpenRocket-{self.openrocket_version}.jar
echo   ork2notebook --filepath your_rocket.ork --ork_jar {self.install_dir}\\OpenRocket-{self.openrocket_version}.jar
''')
            print(f"✅ Created activation script: {activate_script}")
        
        return activate_script
    
    def print_summary(self, jar_path, venv_path, activate_script):
        """Print installation summary and next steps"""
        self.print_step("Installation Complete! 🚀")
        
        print("All dependencies have been installed successfully!")
        print(f"\nInstallation directory: {self.install_dir}")
        print(f"OpenRocket JAR: {jar_path}")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        
        print("\n1. Convert your .ork file to JSON:")
        print(f"   ork2json --filepath your_rocket.ork --ork_jar {jar_path}")
        
        print("\n2. Or convert to a Jupyter notebook:")
        print(f"   ork2notebook --filepath your_rocket.ork --ork_jar {jar_path}")
        
        print("\n3. Check the example script:")
        print(f"   {self.install_dir / 'example_usage.py'}")
        
        print("\n4. Import and use RocketPy in your scripts:")
        print("   from rocketpy import Environment, Rocket, SolidMotor, Flight")
        
        print("\n" + "="*60)
        print("DOCUMENTATION:")
        print("="*60)
        print("RocketPy: https://docs.rocketpy.org/")
        print("RocketSerializer: https://github.com/RocketPy-Team/RocketSerializer")
        
        print("\n✅ You're ready to run simulations!")
    
    def run(self):
        """Run the complete installation process"""
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  RocketPy + RocketSerializer Installer".center(58) + "║")
        print("║" + "  Cross-Platform OpenRocket Simulation Setup".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")
        
        # Check Python version
        self.check_python_version()
        
        # Check and install Java if needed
        if not self.check_java():
            self.install_java()
        
        # Download OpenRocket JAR
        jar_path = self.download_openrocket_jar()
        
        # Create virtual environment
        venv_path = self.create_venv()
        
        # Install Python packages in venv
        self.install_python_packages(venv_path)
        
        # Create config file
        self.create_config_file(jar_path, venv_path)
        
        # Create activation helper scripts
        activate_script = self.create_activation_scripts(venv_path)
        
        # Create example script
        self.create_example_script(venv_path)
        
        # Print summary
        self.print_summary(jar_path, venv_path, activate_script)

if __name__ == "__main__":
    try:
        installer = RocketPyInstaller()
        installer.run()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
