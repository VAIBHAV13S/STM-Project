#!/usr/bin/env python3
"""
Smart Traffic Management System - Main Orchestrator
This script coordinates all components of the traffic management system.
"""

import os
import sys
import time
import json
import subprocess
import threading
import argparse
from datetime import datetime
from pathlib import Path


class TrafficManagementOrchestrator:
    """Main orchestrator for the traffic management system"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.processes = {}
        self.running = False
        
    def start_sumo_simulation(self, route_type="balanced", steps=3600, gui=False):
        """Start SUMO simulation"""
        print("Starting SUMO simulation...")
        
        sumo_dir = self.base_dir / "sumo"
        cmd = [
            sys.executable, 
            str(sumo_dir / "runner.py"),
            "--route", route_type,
            "--steps", str(steps)
        ]
        
        if gui:
            cmd.append("--gui")
        
        process = subprocess.Popen(cmd, cwd=sumo_dir)
        self.processes['sumo'] = process
        return process
    
    def start_ml_training(self):
        """Start ML model training"""
        print("Starting ML model training...")
        
        ml_dir = self.base_dir / "ml"
        cmd = [
            sys.executable,
            str(ml_dir / "train_model.py"),
            "--plot"
        ]
        
        process = subprocess.Popen(cmd, cwd=ml_dir)
        self.processes['ml_training'] = process
        return process
    
    def start_scheduler(self):
        """Start slot scheduler"""
        print("Starting slot scheduler...")
        
        scheduler_dir = self.base_dir / "scheduling"
        cmd = [
            sys.executable,
            str(scheduler_dir / "slot_scheduler.py")
        ]
        
        process = subprocess.Popen(cmd, cwd=scheduler_dir)
        self.processes['scheduler'] = process
        return process
    
    def start_dashboard(self):
        """Start visualization dashboard"""
        print("Starting dashboard...")
        
        viz_dir = self.base_dir / "visualization"
        cmd = [
            "streamlit", "run",
            str(viz_dir / "dashboard.py"),
            "--server.port", "8501",
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(cmd, cwd=viz_dir)
        self.processes['dashboard'] = process
        return process
    
    def start_hardware_simulation(self):
        """Start hardware simulation (sensors and lights)"""
        print("Starting hardware simulation...")
        
        hardware_dir = self.base_dir / "hardware"
        
        # Start sensor simulation
        sensor_cmd = [
            sys.executable,
            str(hardware_dir / "ultrasonic_sensor.py"),
            "--simulation",
            "--duration", "3600"
        ]
        
        sensor_process = subprocess.Popen(sensor_cmd, cwd=hardware_dir)
        self.processes['sensors'] = sensor_process
        
        # Start traffic light simulation
        time.sleep(2)  # Wait a bit before starting lights
        
        lights_cmd = [
            sys.executable,
            str(hardware_dir / "traffic_lights.py"),
            "--simulation",
            "--duration", "3600"
        ]
        
        lights_process = subprocess.Popen(lights_cmd, cwd=hardware_dir)
        self.processes['lights'] = lights_process
        
        return sensor_process, lights_process
    
    def stop_all_processes(self):
        """Stop all running processes"""
        print("Stopping all processes...")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
        
        self.processes.clear()
        self.running = False
    
    def run_full_system(self, duration=3600, with_gui=False):
        """Run the complete traffic management system"""
        print("=" * 60)
        print("SMART TRAFFIC MANAGEMENT SYSTEM - FULL DEPLOYMENT")
        print("=" * 60)
        
        self.running = True
        
        try:
            # Phase 1: Start SUMO simulation
            print("\n[Phase 1] Starting SUMO simulation...")
            self.start_sumo_simulation("balanced", duration, with_gui)
            time.sleep(5)
            
            # Phase 2: Start ML training (if data exists)
            data_dir = self.base_dir / "data"
            if any(data_dir.glob("*.csv")):
                print("\n[Phase 2] Starting ML training...")
                self.start_ml_training()
                time.sleep(10)  # Give training time to start
            
            # Phase 3: Start slot scheduler
            print("\n[Phase 3] Starting slot scheduler...")
            self.start_scheduler()
            time.sleep(2)
            
            # Phase 4: Start dashboard
            print("\n[Phase 4] Starting visualization dashboard...")
            self.start_dashboard()
            time.sleep(5)
            
            # Phase 5: Start hardware simulation
            print("\n[Phase 5] Starting hardware simulation...")
            self.start_hardware_simulation()
            
            print(f"\n🚦 System running! Dashboard available at: http://localhost:8501")
            print(f"⏱️  Running for {duration} seconds...")
            
            # Monitor processes
            start_time = time.time()
            while self.running and (time.time() - start_time) < duration:
                time.sleep(10)
                
                # Check if any critical processes have died
                dead_processes = []
                for name, process in self.processes.items():
                    if process and process.poll() is not None:
                        dead_processes.append(name)
                
                if dead_processes:
                    print(f"⚠️  Dead processes detected: {dead_processes}")
                
                print(f"📊 System status: {len(self.processes) - len(dead_processes)}/{len(self.processes)} processes running")
        
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user...")
        
        except Exception as e:
            print(f"❌ Error in system orchestration: {e}")
        
        finally:
            self.stop_all_processes()
            print("✅ System shutdown complete")
    
    def run_phase_by_phase(self):
        """Run system components phase by phase for demonstration"""
        print("=" * 60)
        print("SMART TRAFFIC MANAGEMENT SYSTEM - PHASE BY PHASE DEMO")
        print("=" * 60)
        
        phases = [
            ("Phase 1: SUMO Simulation", self.demo_phase1),
            ("Phase 2: ML Training", self.demo_phase2),
            ("Phase 3: Slot Scheduling", self.demo_phase3),
            ("Phase 4: Visualization", self.demo_phase4),
            ("Phase 5: Hardware Simulation", self.demo_phase5)
        ]
        
        for phase_name, phase_func in phases:
            print(f"\n{'='*20} {phase_name} {'='*20}")
            try:
                phase_func()
                input("Press Enter to continue to next phase...")
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
                break
        
        self.stop_all_processes()
    
    def demo_phase1(self):
        """Demo Phase 1: SUMO Simulation"""
        print("Running SUMO simulation with data collection...")
        process = self.start_sumo_simulation("rush_hour", 600, False)
        process.wait()
        print("✅ Phase 1 complete - Vehicle data collected")
    
    def demo_phase2(self):
        """Demo Phase 2: ML Training"""
        print("Training intent prediction model...")
        process = self.start_ml_training()
        process.wait()
        print("✅ Phase 2 complete - ML model trained")
    
    def demo_phase3(self):
        """Demo Phase 3: Slot Scheduling"""
        print("Running slot scheduler demonstration...")
        process = self.start_scheduler()
        time.sleep(10)  # Let it run for 10 seconds
        process.terminate()
        print("✅ Phase 3 complete - Slot scheduling demonstrated")
    
    def demo_phase4(self):
        """Demo Phase 4: Visualization"""
        print("Starting visualization dashboard...")
        process = self.start_dashboard()
        print("📊 Dashboard started at http://localhost:8501")
        print("Open the URL in your browser to view the dashboard")
        time.sleep(30)  # Let it run for 30 seconds
        process.terminate()
        print("✅ Phase 4 complete - Dashboard demonstrated")
    
    def demo_phase5(self):
        """Demo Phase 5: Hardware Simulation"""
        print("Running hardware simulation...")
        sensor_proc, lights_proc = self.start_hardware_simulation()
        time.sleep(20)  # Let it run for 20 seconds
        sensor_proc.terminate()
        lights_proc.terminate()
        print("✅ Phase 5 complete - Hardware simulation demonstrated")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("Checking system dependencies...")
        
        required_packages = [
            'traci', 'pandas', 'numpy', 'sklearn', 'streamlit', 'plotly'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {missing_packages}")
            print("Install with: pip install -r requirements.txt")
            return False
        
        print("\n✅ All dependencies satisfied")
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Smart Traffic Management System Orchestrator')
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'demo', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'check'],
                       help='Run mode')
    parser.add_argument('--duration', type=int, default=3600, help='Run duration in seconds')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--base-dir', help='Base directory for the project')
    
    args = parser.parse_args()
    
    orchestrator = TrafficManagementOrchestrator(args.base_dir)
    
    if args.mode == 'check':
        orchestrator.check_dependencies()
    elif args.mode == 'full':
        if orchestrator.check_dependencies():
            orchestrator.run_full_system(args.duration, args.gui)
    elif args.mode == 'demo':
        if orchestrator.check_dependencies():
            orchestrator.run_phase_by_phase()
    elif args.mode.startswith('phase'):
        phase_num = args.mode[-1]
        method_name = f'demo_phase{phase_num}'
        if hasattr(orchestrator, method_name):
            getattr(orchestrator, method_name)()
        else:
            print(f"Invalid phase: {phase_num}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
