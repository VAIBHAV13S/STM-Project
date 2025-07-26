#!/usr/bin/env python3
"""
Enhanced Setup Script for Smart Traffic Management System
Windows-specific improvements and better error handling.
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run command with error handling"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True, result.stdout
        else:
            print(f"❌ {description}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - Timeout")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False, str(e)


def install_dependencies_individually():
    """Install each dependency individually to catch specific errors"""
    print("\nInstalling Python dependencies individually...")
    
    packages = [
        "traci>=1.14.1",
        "scikit-learn>=1.3.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success, output = run_command([sys.executable, "-m", "pip", "install", package])
        if not success:
            failed_packages.append(package)
            print(f"Failed to install {package}")
    
    if failed_packages:
        print(f"\n❌ Failed packages: {failed_packages}")
        print("\nTrying alternative installation methods...")
        
        # Try with --user flag
        for package in failed_packages:
            print(f"Trying --user installation for {package}...")
            success, _ = run_command([sys.executable, "-m", "pip", "install", "--user", package])
            if success:
                failed_packages.remove(package)
    
    return len(failed_packages) == 0


def check_sumo_detailed():
    """Detailed SUMO installation check with Windows-specific paths"""
    print("\nDetailed SUMO installation check...")
    
    # Common Windows SUMO installation paths
    possible_paths = [
        "C:\\Program Files (x86)\\Eclipse\\Sumo",
        "C:\\Program Files\\Eclipse\\Sumo", 
        "C:\\sumo",
        "C:\\Eclipse\\Sumo"
    ]
    
    # Check SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        print(f"SUMO_HOME is set: {sumo_home}")
        possible_paths.insert(0, sumo_home)
    else:
        print("SUMO_HOME not set, checking common locations...")
    
    # Check each possible path
    for path in possible_paths:
        print(f"Checking: {path}")
        sumo_path = Path(path)
        
        if sumo_path.exists():
            print(f"✅ SUMO directory found: {path}")
            
            # Check for bin directory
            bin_path = sumo_path / "bin"
            if bin_path.exists():
                print(f"✅ bin directory found: {bin_path}")
                
                # Check for sumo executable
                sumo_exe = bin_path / "sumo.exe"
                sumo_gui_exe = bin_path / "sumo-gui.exe"
                
                if sumo_exe.exists() or sumo_gui_exe.exists():
                    print(f"✅ SUMO executables found")
                    
                    # Set environment variables if not set
                    if not sumo_home:
                        print(f"Setting SUMO_HOME to {path}")
                        os.environ['SUMO_HOME'] = str(path)
                        
                        # Add to PATH for current session
                        current_path = os.environ.get('PATH', '')
                        if str(bin_path) not in current_path:
                            os.environ['PATH'] = f"{current_path};{bin_path}"
                    
                    return True, path
                else:
                    print(f"❌ SUMO executables not found in {bin_path}")
            else:
                print(f"❌ bin directory not found in {path}")
        else:
            print(f"❌ Directory not found: {path}")
    
    return False, None


def provide_sumo_installation_instructions():
    """Provide detailed SUMO installation instructions for Windows"""
    print("\n" + "="*70)
    print("SUMO INSTALLATION INSTRUCTIONS FOR WINDOWS")
    print("="*70)
    
    print("\n📥 Step 1: Download SUMO")
    print("1. Go to: https://sumo.dlr.de/docs/Installing/Windows_Build.html")
    print("2. Click on 'Windows binaries' link")
    print("3. Download the latest .msi installer file")
    print("   (e.g., sumo-win64-1.19.0.msi)")
    
    print("\n🔧 Step 2: Install SUMO")
    print("1. Right-click the downloaded .msi file")
    print("2. Select 'Run as administrator'")
    print("3. Follow installation wizard")
    print("4. Install to default location (recommended):")
    print("   C:\\Program Files (x86)\\Eclipse\\Sumo")
    
    print("\n🌍 Step 3: Set Environment Variables")
    print("Method A - Using Command Prompt (Run as Administrator):")
    print('   setx SUMO_HOME "C:\\Program Files (x86)\\Eclipse\\Sumo" /M')
    print('   setx PATH "%PATH%;C:\\Program Files (x86)\\Eclipse\\Sumo\\bin" /M')
    
    print("\nMethod B - Using System Properties:")
    print("1. Right-click 'This PC' → Properties")
    print("2. Advanced System Settings → Environment Variables")
    print("3. Under System Variables, click 'New':")
    print("   - Variable name: SUMO_HOME")
    print("   - Variable value: C:\\Program Files (x86)\\Eclipse\\Sumo")
    print("4. Edit 'Path' variable, add:")
    print("   - C:\\Program Files (x86)\\Eclipse\\Sumo\\bin")
    
    print("\n🔄 Step 4: Restart")
    print("1. Close all Command Prompt/PowerShell windows")
    print("2. Open new PowerShell window")
    print("3. Test: echo $env:SUMO_HOME")
    print("4. Test: sumo --version")
    
    print("\n📞 Alternative: Quick Install Script")
    print("Create install_sumo.bat with:")
    print("@echo off")
    print("echo Downloading SUMO...")
    print("powershell -Command \"Invoke-WebRequest -Uri 'https://sourceforge.net/projects/sumo/files/sumo/version_1_19_0/sumo-win64-1.19.0.msi/download' -OutFile 'sumo.msi'\"")
    print("echo Installing SUMO...")
    print("msiexec /i sumo.msi /quiet")
    print('setx SUMO_HOME "C:\\Program Files (x86)\\Eclipse\\Sumo" /M')
    print('setx PATH "%PATH%;C:\\Program Files (x86)\\Eclipse\\Sumo\\bin" /M')
    print("echo Installation complete. Please restart PowerShell.")


def main():
    """Enhanced main setup function"""
    print("="*70)
    print("SMART TRAFFIC MANAGEMENT SYSTEM - ENHANCED SETUP")
    print("="*70)
    
    # Check Python
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check SUMO with detailed analysis
    sumo_found, sumo_path = check_sumo_detailed()
    
    if not sumo_found:
        provide_sumo_installation_instructions()
        print("\n⚠️  Please install SUMO first, then run this script again.")
        return 1
    
    # Install dependencies
    deps_success = install_dependencies_individually()
    
    if not deps_success:
        print("\n❌ Some Python dependencies failed to install")
        print("Try running these commands manually:")
        print("pip install --upgrade pip")
        print("pip install --user -r requirements.txt")
        return 1
    
    # Create directories
    base_dir = Path(__file__).parent
    for dirname in ["data", "logs", "models", "exports"]:
        (base_dir / dirname).mkdir(exist_ok=True)
    
    print("\n🎉 Setup completed successfully!")
    print(f"SUMO found at: {sumo_path}")
    print("\nNext steps:")
    print("1. python setup.py  # Run original setup to verify")
    print("2. python main.py --mode demo")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())