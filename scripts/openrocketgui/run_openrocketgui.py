#!/usr/bin/env python3
"""
OpenRocket GUI Automation Script
Automates interactions with OpenRocket GUI to run simulations programmatically
Uses pyautogui for GUI automation
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path
import json

try:
    import pyautogui
except ImportError:
    print("❌ pyautogui not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
    import pyautogui

# For Linux, also need additional dependencies
if platform.system() == "Linux":
    try:
        import Xlib
    except ImportError:
        print("⚠️  For Linux, you may need: sudo apt install python3-xlib python3-tk scrot")

class OpenRocketAutomation:
    def __init__(self, jar_path, ork_file=None):
        """
        Initialize OpenRocket automation
        
        Args:
            jar_path: Path to OpenRocket JAR file
            ork_file: Optional path to .ork file to open
        """
        self.jar_path = Path(jar_path)
        self.ork_file = Path(ork_file) if ork_file else None
        self.process = None
        self.system = platform.system()
        
        # Timing delays (in seconds) - adjust based on your system speed
        self.launch_delay = 3  # Time to wait for OpenRocket to launch
        self.dialog_delay = 0.5  # Time to wait for dialogs to appear
        self.simulation_delay = 2  # Time to wait for simulation to complete
        
        # Safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.5  # Pause between actions
        
        if not self.jar_path.exists():
            raise FileNotFoundError(f"OpenRocket JAR not found: {self.jar_path}")
        
        if self.ork_file and not self.ork_file.exists():
            raise FileNotFoundError(f"ORK file not found: {self.ork_file}")
    
    def launch_openrocket(self):
        """Launch OpenRocket GUI"""
        print(f"🚀 Launching OpenRocket...")
        
        cmd = ["java", "-jar", str(self.jar_path)]
        
        if self.ork_file:
            cmd.append(str(self.ork_file))
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"⏳ Waiting {self.launch_delay}s for OpenRocket to load...")
        time.sleep(self.launch_delay)
        print("✅ OpenRocket should be open now")
    
    def close_openrocket(self):
        """Close OpenRocket"""
        print("🔒 Closing OpenRocket...")
        
        # Try graceful close with Ctrl+Q or Alt+F4
        if self.system == "Darwin":
            pyautogui.hotkey('command', 'q')
        else:
            pyautogui.hotkey('alt', 'F4')
        
        time.sleep(1)
        
        # Force kill if still running
        if self.process and self.process.poll() is None:
            self.process.terminate()
            time.sleep(1)
            if self.process.poll() is None:
                self.process.kill()
    
    def open_file(self, ork_file):
        """Open an .ork file"""
        print(f"📂 Opening file: {ork_file}")
        
        # Ctrl+O to open file dialog
        if self.system == "Darwin":
            pyautogui.hotkey('command', 'o')
        else:
            pyautogui.hotkey('ctrl', 'o')
        
        time.sleep(self.dialog_delay)
        
        # Type the file path
        pyautogui.write(str(Path(ork_file).absolute()), interval=0.05)
        time.sleep(0.5)
        
        # Press Enter to open
        pyautogui.press('enter')
        time.sleep(1)
        print("✅ File opened")
    
    def run_simulation(self, simulation_name=None):
        """
        Run a simulation
        
        Args:
            simulation_name: Optional name of simulation to run (uses first if None)
        """
        print(f"▶️  Running simulation...")
        
        # Press F5 to run simulation (or use menu: Simulation -> Run simulation)
        pyautogui.press('F5')
        
        time.sleep(self.dialog_delay)
        
        # If multiple simulations exist, select the one we want
        if simulation_name:
            # TODO: Add logic to select specific simulation by name
            pass
        
        # The simulation dialog should appear, press Enter to start
        pyautogui.press('enter')
        
        print(f"⏳ Waiting {self.simulation_delay}s for simulation to complete...")
        time.sleep(self.simulation_delay)
        
        print("✅ Simulation complete")
    
    def export_simulation_data(self, output_path, file_format='csv'):
        """
        Export simulation data
        
        Args:
            output_path: Path to save the exported data
            file_format: Export format ('csv', 'txt')
        """
        print(f"💾 Exporting simulation data to: {output_path}")
        
        # Open export dialog
        # Typically: Right-click on simulation -> Export simulation data
        # Or use menu if available
        
        # This is highly dependent on OpenRocket's UI structure
        # You may need to adjust coordinates based on your screen
        
        # For now, use keyboard shortcuts
        pyautogui.hotkey('ctrl', 'e')  # Might not work, depends on OR version
        
        time.sleep(self.dialog_delay)
        
        # Type output path
        pyautogui.write(str(Path(output_path).absolute()), interval=0.05)
        time.sleep(0.5)
        
        # Select file format if dropdown appears
        if file_format.lower() == 'csv':
            pyautogui.press('down')  # Navigate dropdown
        
        pyautogui.press('enter')
        time.sleep(0.5)
        
        print("✅ Data exported")
    
    def get_simulation_results(self):
        """
        Attempt to read simulation results from OpenRocket window
        This is very difficult without direct API access
        """
        print("⚠️  Direct result extraction from GUI is not reliable")
        print("    Consider using export functionality instead")
        return None
    
    def take_screenshot(self, output_path):
        """Take a screenshot of OpenRocket window"""
        print(f"📸 Taking screenshot: {output_path}")
        screenshot = pyautogui.screenshot()
        screenshot.save(output_path)
        print("✅ Screenshot saved")
    
    def batch_run_simulations(self, ork_files, output_dir):
        """
        Run simulations for multiple .ork files
        
        Args:
            ork_files: List of .ork file paths
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, ork_file in enumerate(ork_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(ork_files)}: {ork_file}")
            print(f"{'='*60}\n")
            
            try:
                # Launch with the file
                self.ork_file = Path(ork_file)
                self.launch_openrocket()
                
                # Run simulation
                self.run_simulation()
                
                # Take screenshot of results
                screenshot_path = output_dir / f"{Path(ork_file).stem}_results.png"
                self.take_screenshot(screenshot_path)
                
                # Export data (if export shortcut works)
                export_path = output_dir / f"{Path(ork_file).stem}_data.csv"
                try:
                    self.export_simulation_data(export_path)
                except Exception as e:
                    print(f"⚠️  Could not export data: {e}")
                
                results.append({
                    'file': str(ork_file),
                    'status': 'success',
                    'screenshot': str(screenshot_path),
                    'export': str(export_path) if export_path.exists() else None
                })
                
            except Exception as e:
                print(f"❌ Error processing {ork_file}: {e}")
                results.append({
                    'file': str(ork_file),
                    'status': 'failed',
                    'error': str(e)
                })
            
            finally:
                # Close OpenRocket
                self.close_openrocket()
                time.sleep(1)
        
        # Save results summary
        summary_path = output_dir / "batch_results.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Batch processing complete!")
        print(f"📊 Summary saved to: {summary_path}")
        
        return results


def print_usage():
    """Print usage instructions"""
    print("""
OpenRocket GUI Automation Script
=================================

This script automates OpenRocket GUI interactions for batch simulations.

IMPORTANT NOTES:
- This uses GUI automation (simulates mouse/keyboard)
- Do NOT move mouse or press keys while script is running
- Move mouse to screen corner to abort (FAILSAFE)
- Adjust timing delays if your computer is slow/fast
- This is a workaround - native API would be more reliable

DEPENDENCIES:
  pip install pyautogui
  
  Linux also needs: sudo apt install python3-xlib python3-tk scrot

USAGE:

1. Single simulation:
   python openrocket_automation.py --jar OpenRocket.jar --ork rocket.ork

2. Batch simulations:
   python openrocket_automation.py --jar OpenRocket.jar --batch *.ork --output results/

3. With custom delays (for slower systems):
   python openrocket_automation.py --jar OpenRocket.jar --ork rocket.ork --launch-delay 5

LIMITATIONS:
- Relies on GUI elements being in expected positions
- Screen resolution and window placement affect reliability
- Can't easily extract numerical results (use exports instead)
- Requires display (won't work on headless servers without Xvfb)

ALTERNATIVES:
- Use RocketPy + RocketSerializer (better for automation)
- Use older OpenRocket versions with documented API
- Request OpenRocket team to restore API support
""")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automate OpenRocket GUI simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--jar', required=True, help='Path to OpenRocket JAR file')
    parser.add_argument('--ork', help='Path to .ork file to simulate')
    parser.add_argument('--batch', nargs='+', help='Multiple .ork files for batch processing')
    parser.add_argument('--output', default='./results', help='Output directory for results')
    parser.add_argument('--launch-delay', type=float, default=3, help='Seconds to wait for launch')
    parser.add_argument('--simulation-delay', type=float, default=2, help='Seconds to wait for simulation')
    parser.add_argument('--screenshot', action='store_true', help='Take screenshot of results')
    
    args = parser.parse_args()
    
    if not args.ork and not args.batch:
        print("❌ Error: Must specify either --ork or --batch")
        print_usage()
        sys.exit(1)
    
    try:
        # Create automation instance
        automation = OpenRocketAutomation(args.jar, args.ork)
        automation.launch_delay = args.launch_delay
        automation.simulation_delay = args.simulation_delay
        
        if args.batch:
            # Batch mode
            print(f"🔄 Batch mode: {len(args.batch)} files")
            automation.batch_run_simulations(args.batch, args.output)
        else:
            # Single file mode
            print(f"🎯 Single file mode: {args.ork}")
            automation.launch_openrocket()
            
            try:
                automation.run_simulation()
                
                if args.screenshot:
                    output_dir = Path(args.output)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = output_dir / f"{Path(args.ork).stem}_results.png"
                    automation.take_screenshot(screenshot_path)
                
                print("\n✅ Simulation complete!")
                print("⚠️  Press Ctrl+C when ready to close OpenRocket")
                
                # Keep window open until user interrupts
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n")
                    
            finally:
                automation.close_openrocket()
    
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
    else:
        main()
