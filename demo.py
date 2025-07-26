#!/usr/bin/env python3
"""
Smart Traffic Management System - Complete Demo
5-Phase Implementation: SUMO Simulation, ML Training, Slot Scheduling, Visualization, Hardware

Run: python demo.py --mode demo
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime


class SmartTrafficDemo:
    """Complete smart traffic management system demonstration"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"📁 Created data directory: {self.data_dir}")
    
    def phase_1_simulation(self):
        """Phase 1: Run SUMO simulation and collect traffic data"""
        print("\n" + "="*80)
        print("🚦 PHASE 1: SUMO Traffic Simulation & Data Collection")
        print("="*80)
        
        print("🔄 Running SUMO simulation with working rush hour pattern...")
        
        # Change to SUMO directory
        sumo_dir = os.path.join(self.base_dir, "sumo")
        original_cwd = os.getcwd()
        os.chdir(sumo_dir)
        
        try:
            # Run simulation
            result = subprocess.run([
                "python", "runner.py", 
                "--route", "working_rush_hour", 
                "--steps", "600",  # 10 minutes simulation
                "--analyze"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ SUMO simulation completed successfully!")
                print("\n📊 Simulation Results:")
                print(result.stdout[-500:])  # Show last 500 chars
                
                # Find the generated data file (using absolute path)
                data_dir = os.path.join(self.base_dir, "data")
                data_files = [f for f in os.listdir(data_dir) if f.startswith("vehicle_data_working_rush_hour")]
                if data_files:
                    latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
                    return os.path.join(data_dir, latest_file)
                else:
                    print("⚠️  No data file found in data directory!")
                    return None
            else:
                print("❌ SUMO simulation failed!")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Simulation timed out after 5 minutes")
            return None
        finally:
            os.chdir(original_cwd)
    
    def phase_2_machine_learning(self, data_file):
        """Phase 2: Train machine learning model for intent prediction"""
        print("\n" + "="*80)
        print("🤖 PHASE 2: Machine Learning Intent Prediction")
        print("="*80)
        
        if not data_file or not os.path.exists(data_file):
            print("❌ No traffic data provided from Phase 1!")
            print("   Looking for existing SUMO data files...")
            
            # Look for existing SUMO data files
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith("vehicle_data_") and not f.startswith("vehicle_data_sample")]
            if data_files:
                latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
                data_file = os.path.join(self.data_dir, latest_file)
                print(f"📊 Found SUMO data: {latest_file}")
            else:
                print("   No SUMO data found. Generating sample data for demonstration...")
                self._generate_sample_data()
                data_file = os.path.join(self.data_dir, "sample_vehicle_data.csv")
            
        print(f"🔄 Training ML model with data from: {os.path.basename(data_file)}")
        
        # Change to ML directory
        ml_dir = os.path.join(self.base_dir, "ml")
        original_cwd = os.getcwd()
        os.chdir(ml_dir)
        
        try:
            # Copy data file to ML directory for training
            import shutil
            local_data_file = "training_data.csv"
            shutil.copy(data_file, local_data_file)
            
            # Run ML training
            result = subprocess.run([
                "python", "train_model.py", 
                "--data-dir", ".",
                "--plot"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ ML model training completed successfully!")
                print("\n📈 Training Results:")
                print(result.stdout[-500:])  # Show last 500 chars
                return "intent_predictor_model.pkl"
            else:
                print("❌ ML model training failed!")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ ML training timed out after 2 minutes")
            return None
        finally:
            os.chdir(original_cwd)
    
    def phase_3_slot_scheduling(self, data_file):
        """Phase 3: Demonstrate slot-based traffic scheduling"""
        print("\n" + "="*80)
        print("📅 PHASE 3: Slot-Based Traffic Scheduling")
        print("="*80)
        
        if not data_file or not os.path.exists(data_file):
            print("❌ No traffic data provided from Phase 1!")
            print("   Looking for existing SUMO data files...")
            
            # Look for existing SUMO data files
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith("vehicle_data_") and not f.startswith("vehicle_data_sample")]
            if data_files:
                latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
                data_file = os.path.join(self.data_dir, latest_file)
                print(f"📊 Found SUMO data: {latest_file}")
            else:
                print("   No SUMO data found. Using sample data for demonstration...")
                self._generate_sample_data()
                data_file = os.path.join(self.data_dir, "sample_vehicle_data.csv")
            
        print("🔄 Running slot-based scheduling simulation...")
        
        # Change to scheduling directory
        scheduling_dir = os.path.join(self.base_dir, "scheduling")
        original_cwd = os.getcwd()
        os.chdir(scheduling_dir)
        
        try:
            # Copy data file for scheduling analysis
            import shutil
            local_data_file = "traffic_data.csv"
            shutil.copy(data_file, local_data_file)
            
            # Run scheduling demo
            result = subprocess.run([
                "python", "slot_scheduler.py", 
                "--data", local_data_file
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Slot scheduling demo completed successfully!")
                print("\n🕐 Scheduling Results:")
                print(result.stdout[-500:])  # Show last 500 chars
                return True
            else:
                print("❌ Slot scheduling demo failed!")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Scheduling demo timed out after 1 minute")
            return False
        finally:
            os.chdir(original_cwd)
    
    def phase_4_visualization(self, data_file):
        """Phase 4: Launch real-time visualization dashboard"""
        print("\n" + "="*80)
        print("📊 PHASE 4: Real-Time Visualization Dashboard")
        print("="*80)
        
        print("🔄 Preparing dashboard...")
        
        # Change to visualization directory
        viz_dir = os.path.join(self.base_dir, "visualization")
        original_cwd = os.getcwd()
        os.chdir(viz_dir)
        
        try:
            if data_file and os.path.exists(data_file):
                # Copy data file for dashboard
                import shutil
                shutil.copy(data_file, "dashboard_data.csv")
                print(f"📊 Using data: {os.path.basename(data_file)}")
            else:
                print("📊 Using default sample data")
            
            print("\n🌐 Starting Streamlit dashboard...")
            print("   Dashboard URL: http://localhost:8501")
            print("   Press Ctrl+C to stop the dashboard")
            
            # Launch Streamlit dashboard
            result = subprocess.run([
                "streamlit", "run", "dashboard.py", 
                "--server.headless", "false",
                "--server.port", "8501"
            ], timeout=30)  # Run for 30 seconds in demo mode
            
            return True
            
        except subprocess.TimeoutExpired:
            print("\n✅ Dashboard demo completed (30 seconds)")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to launch Streamlit dashboard!")
            print("   Make sure Streamlit is installed: pip install streamlit")
            return False
        except KeyboardInterrupt:
            print("\n✅ Dashboard stopped by user")
            return True
        finally:
            os.chdir(original_cwd)
    
    def phase_5_hardware_simulation(self):
        """Phase 5: Hardware prototype simulation"""
        print("\n" + "="*80)
        print("🔧 PHASE 5: Hardware Prototype Simulation")
        print("="*80)
        
        print("🔄 Running hardware simulation (Raspberry Pi GPIO emulation)...")
        
        # Change to hardware directory
        hardware_dir = os.path.join(self.base_dir, "hardware")
        original_cwd = os.getcwd()
        os.chdir(hardware_dir)
        
        try:
            # Run hardware simulation
            result = subprocess.run([
                "python", "traffic_light_controller.py", 
                "--demo", "--duration", "30"
            ], capture_output=True, text=True, timeout=45)
            
            if result.returncode == 0:
                print("✅ Hardware simulation completed successfully!")
                print("\n🚥 Hardware Results:")
                print(result.stdout[-500:])  # Show last 500 chars
                return True
            else:
                print("❌ Hardware simulation failed!")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Hardware simulation timed out")
            return False
        finally:
            os.chdir(original_cwd)
    
    def _generate_sample_data(self):
        """Generate sample vehicle data for demonstration"""
        import pandas as pd
        import numpy as np
        
        # Generate sample vehicle data
        np.random.seed(42)
        n_records = 1000
        
        data = {
            'timestamp': [datetime.now().isoformat()] * n_records,
            'step': range(n_records),
            'vehicle_id': [f"vehicle_{i%50}" for i in range(n_records)],
            'x': np.random.uniform(50, 150, n_records),
            'y': np.random.uniform(50, 150, n_records),
            'speed': np.random.exponential(8, n_records),
            'acceleration': np.random.normal(0, 2, n_records),
            'lane_id': np.random.choice(['north_in_0', 'south_in_0', 'east_in_0', 'west_in_0'], n_records),
            'lane_position': np.random.uniform(0, 100, n_records),
            'route_id': np.random.choice(['route_ns', 'route_ew', 'route_ne', 'route_nw'], n_records),
            'route_edges': ['north_in|south_out'] * (n_records//3) + ['east_in|west_out'] * (n_records//3) + ['north_in|east_out'] * (n_records - 2*(n_records//3)),
            'intent': np.random.choice(['straight', 'left', 'right'], n_records, p=[0.5, 0.25, 0.25]),
            'distance_to_intersection': np.random.uniform(10, 100, n_records),
            'waiting_time': np.random.exponential(5, n_records),
            'angle': np.random.uniform(0, 360, n_records),
            'vehicle_type': ['car'] * n_records
        }
        
        df = pd.DataFrame(data)
        sample_file = os.path.join(self.data_dir, "sample_vehicle_data.csv")
        df.to_csv(sample_file, index=False)
        print(f"📊 Generated sample data: {sample_file}")
    
    def run_full_demo(self):
        """Run complete 5-phase demonstration"""
        print("🚀 SMART TRAFFIC MANAGEMENT SYSTEM - COMPLETE DEMO")
        print("=" * 80)
        print("This demonstration showcases all 5 phases of the system:")
        print("1. 🚦 SUMO Traffic Simulation")
        print("2. 🤖 Machine Learning Training")
        print("3. 📅 Slot-Based Scheduling")
        print("4. 📊 Real-Time Visualization")
        print("5. 🔧 Hardware Prototype")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # Phase 1: SUMO Simulation
        data_file = self.phase_1_simulation()
        
        # Phase 2: Machine Learning
        model_file = self.phase_2_machine_learning(data_file)
        
        # Phase 3: Slot Scheduling
        scheduling_success = self.phase_3_slot_scheduling(data_file)
        
        # Phase 5: Hardware Simulation
        hardware_success = self.phase_5_hardware_simulation()
        
        # Summary
        print("\n" + "="*80)
        print("🎯 DEMO SUMMARY")
        print("="*80)
        
        phases_status = [
            ("Phase 1 - SUMO Simulation", "✅" if data_file else "❌"),
            ("Phase 2 - ML Training", "✅" if model_file else "❌"),
            ("Phase 3 - Slot Scheduling", "✅" if scheduling_success else "❌"),
            ("Phase 4 - Visualization", "⏳ Ready to launch"),
            ("Phase 5 - Hardware Simulation", "✅" if hardware_success else "❌")
        ]
        
        for phase, status in phases_status:
            print(f"{status} {phase}")
        
        total_time = datetime.now() - start_time
        print(f"\n⏱️  Total demo time: {total_time.total_seconds():.1f} seconds")
        
        # Launch dashboard
        if input("\n🌐 Launch visualization dashboard? (y/n): ").lower().startswith('y'):
            self.phase_4_visualization(data_file)
        
        print("\n🎉 Demo completed! Thank you for exploring the Smart Traffic Management System!")
        print("\n📝 System Overview:")
        print("   • SUMO microsimulation with TraCI integration")
        print("   • Machine learning for vehicle intent prediction")
        print("   • Slot-based conflict-free scheduling algorithm")
        print("   • Real-time Streamlit visualization dashboard")
        print("   • Raspberry Pi GPIO hardware prototype simulation")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Traffic Management System Demo')
    parser.add_argument('--mode', default='demo', choices=['demo', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5'],
                       help='Demo mode')
    
    args = parser.parse_args()
    
    demo = SmartTrafficDemo()
    
    if args.mode == 'demo':
        demo.run_full_demo()
    elif args.mode == 'phase1':
        demo.phase_1_simulation()
    elif args.mode == 'phase2':
        demo.phase_2_machine_learning(None)
    elif args.mode == 'phase3':
        demo.phase_3_slot_scheduling(None)
    elif args.mode == 'phase4':
        demo.phase_4_visualization(None)
    elif args.mode == 'phase5':
        demo.phase_5_hardware_simulation()


if __name__ == "__main__":
    main()
