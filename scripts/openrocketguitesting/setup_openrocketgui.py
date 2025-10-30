#!/usr/bin/env python3
"""
OpenRocket GUI Automation Installer
Installs dependencies for automating OpenRocket GUI interactions
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class GUIAutomationInstaller:
    def __init__(self):
        self.system = platform.system()
        self.install_dir = Path.home() / ".rocketpy_deps"
        
    def print_step(self, message):
        """Print a formatted step message"""
        print(f"\n{'='*60}")
        print(f"  {message}")
        print(f"{'='*60}\n")
    
    def check_display(self):
        """Check if display is available (required for GUI automation)"""
        self.print_step("Checking display availability")
        
        if self.system == "Linux":
            if not os.environ.get('DISPLAY'):
                print("❌ No DISPLAY environment variable found")
                print("   GUI automation requires a graphical display")
                print("   For headless servers, consider using Xvfb")
                return False
        
        print("✅ Display available")
        return True
    
    def create_venv(self):
        """Create a virtual environment"""
        self.print_step("Setting up Python virtual environment")
        
        venv_path = self.install_dir / "gui_venv"
        pip_path = venv_path / ("Scripts/pip.exe" if self.system == "Windows" else "bin/pip")
        
        # Check if venv exists and is valid
        if venv_path.exists():
            if pip_path.exists():
                print(f"✅ Virtual environment already exists at {venv_path}")
                return venv_path
            else:
                print(f"⚠️  Existing venv is incomplete, recreating...")
                import shutil
                shutil.rmtree(venv_path)
        
        print(f"Creating virtual environment at {venv_path}...")
        try:
            import venv
            venv.create(venv_path, with_pip=True)
            
            if not pip_path.exists():
                raise Exception("pip was not installed in the virtual environment")
            
            print(f"✅ Virtual environment created successfully")
            return venv_path
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            if self.system == "Linux":
                print(f"\nTry: sudo apt install python3.{sys.version_info.minor}-venv")
            sys.exit(1)
    
    def get_venv_python(self, venv_path):
        """Get path to venv Python"""
        if self.system == "Windows":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"
    
    def get_venv_pip(self, venv_path):
        """Get path to venv pip"""
        if self.system == "Windows":
            return venv_path / "Scripts" / "pip.exe"
        return venv_path / "bin" / "pip"
    
    def install_python_packages(self, venv_path):
        """Install required Python packages in venv"""
        self.print_step("Installing Python packages in virtual environment")
        
        pip_path = self.get_venv_pip(venv_path)
        
        packages = [
            "pyautogui",
            "pillow",  # For screenshot support
        ]
        
        if self.system == "Linux":
            packages.append("python-xlib")  # For Linux GUI automation
        
        # Upgrade pip first
        print("Upgrading pip...")
        try:
            subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        except subprocess.CalledProcessError:
            print("⚠️  Could not upgrade pip, continuing...")
        
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([str(pip_path), "install", package])
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Could not install {package}: {e}")
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        if self.system != "Linux":
            print("✅ No additional system dependencies needed")
            return
        
        self.print_step("Checking system dependencies (Linux)")
        
        required_packages = {
            'scrot': 'Screenshot utility',
            'python3-tk': 'Tkinter support',
            'python3-dev': 'Python development headers'
        }
        
        missing = []
        
        for package in required_packages.keys():
            try:
                result = subprocess.run(
                    ['dpkg', '-l', package],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    missing.append(package)
            except FileNotFoundError:
                # dpkg not available, probably not Debian/Ubuntu
                print("⚠️  Could not check for system packages (non-Debian system)")
                break
        
        if missing:
            print(f"⚠️  Missing system packages: {', '.join(missing)}")
            print("\nTo install them, run:")
            print(f"  sudo apt install {' '.join(missing)}")
            
            response = input("\nWould you like to see alternative installation methods? (y/n): ")
            if response.lower() == 'y':
                print("\nFedora/RHEL:")
                print("  sudo dnf install scrot python3-tkinter python3-devel")
                print("\nArch Linux:")
                print("  sudo pacman -S scrot tk python")
        else:
            print("✅ All system dependencies installed")
    
    def create_example_automation_script(self, venv_path):
        """Create example automation scripts"""
        self.print_step("Creating example automation scripts")
        
        self.install_dir.mkdir(parents=True, exist_ok=True)
        venv_python = self.get_venv_python(venv_path)
        
        if self.system == "Windows":
            # Windows batch files
            # Example 1: Simple simulation
            example1 = self.install_dir / "example_single_sim.bat"
            with open(example1, 'w') as f:
                f.write(f'''@echo off
REM Example: Run a single simulation with OpenRocket GUI automation

set OPENROCKET_JAR={self.install_dir}\\OpenRocket-23.09.jar
set ORK_FILE=your_rocket.ork
set OUTPUT_DIR=.\\simulation_results
set VENV_PYTHON={venv_python}

"%VENV_PYTHON%" openrocket_automation.py ^
    --jar "%OPENROCKET_JAR%" ^
    --ork "%ORK_FILE%" ^
    --output "%OUTPUT_DIR%" ^
    --screenshot

echo Simulation complete! Check %OUTPUT_DIR% for results.
pause
''')
            
            # Example 2: Batch simulations
            example2 = self.install_dir / "example_batch_sim.bat"
            with open(example2, 'w') as f:
                f.write(f'''@echo off
REM Example: Run batch simulations with OpenRocket GUI automation

set OPENROCKET_JAR={self.install_dir}\\OpenRocket-23.09.jar
set OUTPUT_DIR=.\\batch_results
set VENV_PYTHON={venv_python}

REM Run simulations for all .ork files in current directory
"%VENV_PYTHON%" openrocket_automation.py ^
    --jar "%OPENROCKET_JAR%" ^
    --batch *.ork ^
    --output "%OUTPUT_DIR%" ^
    --launch-delay 5 ^
    --simulation-delay 3

echo Batch processing complete! Check %OUTPUT_DIR% for results.
pause
''')
        else:
            # Unix shell scripts (Linux/Mac)
            # Example 1: Simple simulation
            example1 = self.install_dir / "example_single_sim.sh"
            with open(example1, 'w') as f:
                f.write(f'''#!/bin/bash
# Example: Run a single simulation with OpenRocket GUI automation

OPENROCKET_JAR="{self.install_dir}/OpenRocket-23.09.jar"
ORK_FILE="your_rocket.ork"
OUTPUT_DIR="./simulation_results"
VENV_PYTHON="{venv_python}"

"$VENV_PYTHON" openrocket_automation.py \\
    --jar "$OPENROCKET_JAR" \\
    --ork "$ORK_FILE" \\
    --output "$OUTPUT_DIR" \\
    --screenshot

echo "Simulation complete! Check $OUTPUT_DIR for results."
''')
            os.chmod(example1, 0o755)
            
            # Example 2: Batch simulations
            example2 = self.install_dir / "example_batch_sim.sh"
            with open(example2, 'w') as f:
                f.write(f'''#!/bin/bash
# Example: Run batch simulations with OpenRocket GUI automation

OPENROCKET_JAR="{self.install_dir}/OpenRocket-23.09.jar"
OUTPUT_DIR="./batch_results"
VENV_PYTHON="{venv_python}"

# Run simulations for all .ork files in current directory
"$VENV_PYTHON" openrocket_automation.py \\
    --jar "$OPENROCKET_JAR" \\
    --batch *.ork \\
    --output "$OUTPUT_DIR" \\
    --launch-delay 5 \\
    --simulation-delay 3

echo "Batch processing complete! Check $OUTPUT_DIR for results."
''')
            os.chmod(example2, 0o755)
        
        print(f"✅ Example scripts created:")
        print(f"   {example1}")
        print(f"   {example2}")
    
    def create_safety_guide(self):
        """Create a safety guide document"""
        self.print_step("Creating safety guide")
        
        guide_path = self.install_dir / "GUI_AUTOMATION_SAFETY.txt"
        
        with open(guide_path, 'w') as f:
            f.write("""
╔══════════════════════════════════════════════════════════════╗
║                  GUI AUTOMATION SAFETY GUIDE                  ║
╚══════════════════════════════════════════════════════════════╝

IMPORTANT: GUI automation simulates keyboard and mouse inputs!

⚠️  CRITICAL SAFETY RULES:

1. DO NOT MOVE MOUSE during script execution
   - The script needs precise control
   - Moving mouse can cause clicks in wrong places

2. DO NOT TYPE during script execution
   - Keystrokes will interfere with automation
   - Could cause unexpected behavior

3. EMERGENCY STOP: Move mouse to screen corner
   - PyAutoGUI FAILSAFE is enabled
   - Moving to corner immediately stops script

4. CLOSE OTHER APPLICATIONS
   - Focus changes can break automation
   - Notifications can interfere

5. DISABLE SCREEN SAVERS
   - Screen lock will break automation
   - Set system to stay awake

6. TEST WITH ONE FILE FIRST
   - Don't start with batch processing
   - Verify timing delays work for your system

⚙️  CONFIGURATION TIPS:

- Slow computer? Increase --launch-delay and --simulation-delay
- Fast computer? Decrease delays for faster processing
- Multiple monitors? Ensure OpenRocket opens on primary display

🐛 TROUBLESHOOTING:

Problem: OpenRocket opens but script fails
Solution: Increase timing delays

Problem: Clicks happen in wrong places  
Solution: Ensure OpenRocket window is maximized and in standard position

Problem: Script hangs
Solution: Move mouse to corner to abort, then adjust delays

Problem: Works manually but not in batch
Solution: Add longer delays between files

📝 LIMITATIONS:

- Requires active display (no headless operation without Xvfb)
- Screen resolution affects button positions
- Different OS versions may have different UI layouts
- Can't extract numerical results reliably
- Less reliable than native API

🔄 BETTER ALTERNATIVES:

If GUI automation is too unreliable, consider:
1. RocketPy + RocketSerializer (for supported designs)
2. OpenRocket older versions with API support
3. Request OpenRocket team to restore API

═══════════════════════════════════════════════════════════════

For more help, see:
- PyAutoGUI docs: https://pyautogui.readthedocs.io/
- OpenRocket forum: https://www.rocketryforum.com/forums/openrocket.142/
""")
        
        print(f"✅ Safety guide created: {guide_path}")
        print("\n⚠️  PLEASE READ THE SAFETY GUIDE BEFORE USING!")
    
    def print_summary(self, venv_path):
        """Print installation summary"""
        self.print_step("Installation Complete! 🎮")
        
        venv_python = self.get_venv_python(venv_path)
        
        print("GUI automation dependencies installed!")
        print(f"\nVirtual environment: {venv_path}")
        print(f"Python executable: {venv_python}")
        
        print("\n" + "="*60)
        print("⚠️  IMPORTANT WARNINGS:")
        print("="*60)
        print("""
1. This approach uses GUI automation - it's FRAGILE and UNRELIABLE
2. Read the safety guide before use
3. DO NOT move mouse or type during execution
4. Move mouse to corner to abort (FAILSAFE)
5. This is a WORKAROUND - native API would be much better
""")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print(f"\n1. Read the safety guide:")
        print(f"   cat {self.install_dir}/GUI_AUTOMATION_SAFETY.txt")
        
        print("\n2. Get the automation script from the artifact above")
        print("   Save it as: openrocket_automation.py")
        
        print("\n3. Test with a single simulation:")
        print(f"   {venv_python} openrocket_automation.py --jar OpenRocket.jar --ork test.ork")
        
        print("\n4. If it works, try batch processing:")
        print(f"   {venv_python} openrocket_automation.py --jar OpenRocket.jar --batch *.ork --output results/")
        
        print("\n" + "="*60)
        print("RECOMMENDED ALTERNATIVE:")
        print("="*60)
        print("Consider using RocketPy + RocketSerializer instead:")
        print("  - More reliable")
        print("  - Doesn't require display")
        print("  - True programmatic control")
        print("  - See: python setup_sim.py")
        
        print("\n" + "="*60)
    
    def run(self):
        """Run installation"""
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  OpenRocket GUI Automation Installer".center(58) + "║")
        print("║" + "  (Experimental/Fragile Approach)".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")
        
        # Check display
        if not self.check_display():
            print("\n⚠️  Cannot proceed without display")
            sys.exit(1)
        
        # Create virtual environment
        venv_path = self.create_venv()
        
        # Install Python packages in venv
        self.install_python_packages(venv_path)
        
        # Check/install system dependencies
        self.install_system_dependencies()
        
        # Create examples
        self.create_example_automation_script(venv_path)
        
        # Create safety guide
        self.create_safety_guide()
        
        # Print summary
        self.print_summary(venv_path)


if __name__ == "__main__":
    try:
        installer = GUIAutomationInstaller()
        installer.run()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
