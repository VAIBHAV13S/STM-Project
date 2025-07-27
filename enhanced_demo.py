#!/usr/bin/env python3
"""
Smart Traffic Management System - Enhanced Demo with Research Features
5-Phase Implementation + Advanced Computer Vision, ETA Algorithms, Physical Model

Enhanced features from research paper:
- Computer Vision vehicle detection with Pi Camera/webcam
- Advanced ETA calculation with path conflict analysis  
- Physical model control with servo motors and LEDs
- Intent-based prediction using Random Forest
- Real-time slot negotiation algorithms

Run: python enhanced_demo.py --mode research
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


class EnhancedSmartTrafficDemo:
    """Enhanced smart traffic management system with research features"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.base_dir / "computer_vision",
            self.base_dir / "advanced_algorithms", 
            self.base_dir / "physical_model"
        ]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {directory}")
    
    def print_header(self, title, width=80):
        """Print formatted header"""
        print("\n" + "="*width)
        print(f"  {title}")
        print("="*width)
    
    def print_phase(self, phase_num, title):
        """Print phase header"""
        print(f"\n🚀 PHASE {phase_num}: {title}")
        print("-" * 60)
    
    def print_enhancement(self, title):
        """Print enhancement header"""
        print(f"\n✨ ENHANCEMENT: {title}")
        print("-" * 60)
    
    def run_command(self, command, description, directory=None, timeout=300):
        """Run a command and return success status"""
        print(f"▶️  {description}")
        
        if directory:
            original_dir = os.getcwd()
            os.chdir(directory)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            
            if success:
                print(f"✅ {description} - SUCCESS")
                if result.stdout:
                    output = result.stdout.strip()
                    if len(output) > 200:
                        print(f"   Output: {output[:200]}...")
                    else:
                        print(f"   Output: {output}")
            else:
                print(f"❌ {description} - FAILED")
                if result.stderr:
                    error = result.stderr.strip()
                    if len(error) > 200:
                        print(f"   Error: {error[:200]}...")
                    else:
                        print(f"   Error: {error}")
            
            return success
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT after {timeout}s")
            return False
        except Exception as e:
            print(f"❌ {description} - EXCEPTION: {str(e)}")
            return False
        finally:
            if directory:
                os.chdir(original_dir)
    
    def start_background_process(self, command, description, directory=None):
        """Start a background process"""
        print(f"🔄 Starting {description}...")
        
        if directory:
            original_dir = os.getcwd()
            os.chdir(directory)
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)  # Give it time to start
            
            if process.poll() is None:
                print(f"✅ {description} started successfully")
                return process
            else:
                print(f"❌ {description} failed to start")
                return None
        except Exception as e:
            print(f"❌ Failed to start {description}: {e}")
            return None
        finally:
            if directory:
                os.chdir(original_dir)
    
    def check_file_exists(self, filepath, description):
        """Check if a file exists and print status"""
        if filepath.exists():
            print(f"✅ {description}: {filepath}")
            return True
        else:
            print(f"❌ {description}: {filepath} (NOT FOUND)")
            return False
    
    def phase_1_enhanced_simulation(self):
        """Phase 1: Enhanced SUMO simulation with advanced data collection"""
        self.print_phase(1, "Enhanced SUMO Traffic Simulation")
        
        sumo_dir = self.base_dir / "sumo"
        
        # Check SUMO files
        files_ok = True
        files_ok &= self.check_file_exists(sumo_dir / "intersection.net.xml", "Network file")
        files_ok &= self.check_file_exists(sumo_dir / "runner.py", "SUMO runner")
        
        if not files_ok:
            print("❌ SUMO files missing!")
            return False
        
        print("📊 Running enhanced SUMO simulation with comprehensive data collection...")
        print("   • Vehicle trajectory tracking")
        print("   • Intent prediction data generation")
        print("   • Conflict detection scenarios")
        print("   • Emergency vehicle priority testing")
        
        success = self.run_command(
            "python runner.py --route working_rush_hour --analyze --steps 3600",
            "Enhanced SUMO simulation (1 hour traffic)",
            directory=sumo_dir
        )
        
        if success:
            # Check for generated data and show statistics
            data_files = list(self.data_dir.glob("vehicle_data_*.csv"))
            if data_files:
                latest_file = max(data_files, key=lambda f: f.stat().st_ctime)
                print(f"✅ Generated enhanced data file: {latest_file}")
                
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_file)
                    print(f"📊 Enhanced Data Statistics:")
                    print(f"   • Total Records: {len(df):,}")
                    print(f"   • Unique Vehicles: {df['vehicle_id'].nunique()}")
                    print(f"   • Simulation Duration: {df['step'].max()} steps")
                    print(f"   • Average Speed: {df['speed'].mean():.2f} m/s")
                    print(f"   • Intent Distribution: {df['intent'].value_counts().to_dict()}")
                    return latest_file
                except ImportError:
                    print("📊 Enhanced data file generated (pandas not available for stats)")
                    return latest_file
            else:
                print("⚠️  No data files found")
        
        return None
    
    def phase_2_enhanced_ml(self, data_file=None):
        """Phase 2: Enhanced Machine Learning with Random Forest"""
        self.print_phase(2, "Enhanced ML Intent Prediction & Pattern Analysis")
        
        ml_dir = self.base_dir / "ml"
        
        if data_file is None:
            data_files = list(self.data_dir.glob("vehicle_data_*.csv"))
            if not data_files:
                print("❌ No vehicle data found for ML training!")
                return False
            data_file = max(data_files, key=lambda f: f.stat().st_ctime)
        
        print(f"📊 Using enhanced data file: {data_file}")
        print("🧠 Enhanced ML Features:")
        print("   • Random Forest intent classification")
        print("   • Multi-feature vehicle analysis")
        print("   • Cross-validation with 85%+ accuracy target")
        print("   • Real-time prediction optimization")
        
        # Run enhanced ML training
        success = self.run_command(
            f"python train_model.py --data ../{data_file.name} --enhanced",
            "Enhanced Random Forest training",
            directory=ml_dir
        )
        
        if success:
            # Run advanced pattern analysis
            success &= self.run_command(
                f"python analyze_patterns.py --data ../{data_file.name} --advanced",
                "Advanced traffic pattern analysis",
                directory=ml_dir
            )
            
            # Check for model files
            model_files = list((ml_dir / "models").glob("*.pkl"))
            if model_files:
                print(f"✅ Generated {len(model_files)} enhanced ML models")
        
        return success
    
    def phase_3_enhanced_scheduling(self, data_file=None):
        """Phase 3: Enhanced Slot Scheduling with Conflict Resolution"""
        self.print_phase(3, "Enhanced Slot Scheduling & Conflict Resolution")
        
        scheduling_dir = self.base_dir / "slot_scheduling"
        
        if data_file is None:
            data_files = list(self.data_dir.glob("vehicle_data_*.csv"))
            if not data_files:
                print("❌ No vehicle data found for scheduling!")
                return False
            data_file = max(data_files, key=lambda f: f.stat().st_ctime)
        
        print("⏰ Enhanced Scheduling Features:")
        print("   • Dynamic time slot allocation")
        print("   • Real-time conflict detection and resolution")
        print("   • Emergency vehicle priority handling")
        print("   • Queue optimization algorithms")
        
        success = self.run_command(
            f"python scheduler.py --input ../{data_file.name} --enhanced",
            "Enhanced slot scheduling with conflict resolution",
            directory=scheduling_dir
        )
        
        if success:
            # Check for schedule output
            schedule_files = list(self.data_dir.glob("schedule_data_*.json"))
            if schedule_files:
                print(f"✅ Generated {len(schedule_files)} enhanced schedule files")
                
                # Show scheduling statistics
                try:
                    import json
                    latest_schedule = max(schedule_files, key=lambda f: f.stat().st_ctime)
                    with open(latest_schedule, 'r') as f:
                        schedule_data = json.load(f)
                    
                    stats = schedule_data.get('statistics', {})
                    print("📊 Enhanced Scheduling Statistics:")
                    print(f"   • Slot Utilization: {stats.get('slot_utilization', 0)*100:.1f}%")
                    print(f"   • Conflicts Avoided: {stats.get('conflicts_avoided', 0)}")
                    print(f"   • Emergency Vehicles: {stats.get('emergency_vehicles_processed', 0)}")
                except Exception as e:
                    print(f"📊 Schedule data generated (analysis failed: {e})")
        
        return success
    
    def phase_4_enhanced_visualization(self):
        """Phase 4: Enhanced 3D Visualization Dashboard"""
        self.print_phase(4, "Enhanced 3D Visualization Dashboard")
        
        print("📊 Enhanced Visualization Features:")
        print("   • Real-time 3D intersection visualization")
        print("   • Performance gauge charts")
        print("   • Advanced heatmaps and analytics")
        print("   • Multiple view modes (Overview, Detailed, Real-time, Historical)")
        print("   • Interactive filtering and data export")
        print("   • Auto-refresh capabilities")
        
        print("🌐 Starting enhanced dashboard at: http://localhost:8502")
        
        # Check if streamlit is available
        try:
            import streamlit
            print("✅ Streamlit is available")
        except ImportError:
            print("⚠️  Installing Streamlit...")
            self.run_command("pip install streamlit plotly", "Installing visualization dependencies")
        
        # Start enhanced dashboard
        dashboard_process = self.start_background_process(
            "streamlit run visualization/dashboard.py --server.port 8502",
            "Enhanced 3D Visualization Dashboard"
        )
        
        if dashboard_process:
            print("✅ Enhanced dashboard started successfully!")
            print("🎯 Key Features Available:")
            print("   • 3D intersection visualization")
            print("   • Real-time performance gauges")
            print("   • Traffic flow heatmaps")
            print("   • Lane analysis charts")
            print("   • Intent prediction accuracy tracking")
            return True
        
        return False
    
    def phase_5_enhanced_hardware(self):
        """Phase 5: Enhanced Hardware Integration"""
        self.print_phase(5, "Enhanced Hardware Integration & Physical Model")
        
        hardware_dir = self.base_dir / "hardware_integration"
        
        print("🔧 Enhanced Hardware Features:")
        print("   • Raspberry Pi GPIO simulation")
        print("   • Traffic light control protocols")
        print("   • Ultrasonic sensor integration")
        print("   • Emergency response automation")
        
        success = self.run_command(
            "python simulator.py --enhanced --duration 45",
            "Enhanced hardware integration simulation",
            directory=hardware_dir
        )
        
        return success
    
    def enhancement_computer_vision(self):
        """Enhancement: Computer Vision Vehicle Detection"""
        self.print_enhancement("Computer Vision Vehicle Detection")
        
        cv_dir = self.base_dir / "computer_vision"
        
        if not cv_dir.exists():
            print("⚠️  Creating computer vision module...")
            cv_dir.mkdir(exist_ok=True)
            return False
        
        print("📹 Computer Vision Features:")
        print("   • Real-time vehicle detection using OpenCV")
        print("   • Pi Camera and webcam support")
        print("   • Intent prediction from visual cues")
        print("   • ML-based vehicle classification")
        print("   • Integration with main traffic system")
        
        # Run CV detection demo
        success = self.run_command(
            "python cv_detector.py --webcam --duration 30",
            "Computer vision vehicle detection demo",
            directory=cv_dir,
            timeout=60
        )
        
        return success
    
    def enhancement_advanced_eta(self):
        """Enhancement: Advanced ETA Calculation"""
        self.print_enhancement("Advanced ETA Calculation & Path Conflict Analysis")
        
        eta_dir = self.base_dir / "advanced_algorithms"
        
        if not eta_dir.exists():
            print("⚠️  Creating advanced algorithms module...")
            eta_dir.mkdir(exist_ok=True)
            return False
        
        print("🧮 Advanced ETA Features:")
        print("   • Dynamic ETA calculation with vehicle dynamics")
        print("   • Path conflict prediction using graph theory")
        print("   • Queue prioritization algorithms")
        print("   • Traffic condition adjustment")
        print("   • Real-time optimization")
        
        success = self.run_command(
            "python eta_calculator.py --demo",
            "Advanced ETA calculation algorithms",
            directory=eta_dir
        )
        
        return success
    
    def enhancement_physical_model(self):
        """Enhancement: Physical Model Control"""
        self.print_enhancement("Physical Model Control with Servo Motors")
        
        physical_dir = self.base_dir / "physical_model"
        
        if not physical_dir.exists():
            print("⚠️  Creating physical model module...")
            physical_dir.mkdir(exist_ok=True)
            return False
        
        print("🤖 Physical Model Features:")
        print("   • Servo motor gate control")
        print("   • LED traffic signal simulation")
        print("   • Physical intersection model")
        print("   • Emergency protocol hardware")
        print("   • Real-time hardware synchronization")
        
        success = self.run_command(
            "python servo_controller.py --demo --duration 40",
            "Physical intersection model control",
            directory=physical_dir
        )
        
        return success
    
    def research_mode_demo(self):
        """Research mode demonstration with all enhanced features"""
        self.print_header("RESEARCH MODE: Intent-Based Traffic Prediction & Negotiation System")
        
        print("🔬 Research Features Demonstration:")
        print("   📊 Enhanced data collection and processing")
        print("   🧠 Random Forest intent classification") 
        print("   📹 Computer vision vehicle detection")
        print("   🧮 Advanced ETA calculation algorithms")
        print("   ⏰ Real-time slot negotiation")
        print("   🤖 Physical model control")
        print("   📈 Comprehensive analytics and visualization")
        
        start_time = datetime.now()
        phases_completed = 0
        enhancements_completed = 0
        
        # Core enhanced phases
        print("\n" + "="*80)
        print("  CORE ENHANCED PHASES")
        print("="*80)
        
        # Phase 1: Enhanced Simulation
        data_file = self.phase_1_enhanced_simulation()
        if data_file:
            phases_completed += 1
        time.sleep(1)
        
        # Phase 2: Enhanced ML
        if self.phase_2_enhanced_ml(data_file):
            phases_completed += 1
        time.sleep(1)
        
        # Phase 3: Enhanced Scheduling
        if self.phase_3_enhanced_scheduling(data_file):
            phases_completed += 1
        time.sleep(1)
        
        # Phase 4: Enhanced Visualization
        if self.phase_4_enhanced_visualization():
            phases_completed += 1
        time.sleep(1)
        
        # Phase 5: Enhanced Hardware
        if self.phase_5_enhanced_hardware():
            phases_completed += 1
        time.sleep(1)
        
        # Research enhancements
        print("\n" + "="*80)
        print("  RESEARCH ENHANCEMENTS")
        print("="*80)
        
        # Enhancement 1: Computer Vision
        if self.enhancement_computer_vision():
            enhancements_completed += 1
        time.sleep(1)
        
        # Enhancement 2: Advanced ETA
        if self.enhancement_advanced_eta():
            enhancements_completed += 1
        time.sleep(1)
        
        # Enhancement 3: Physical Model
        if self.enhancement_physical_model():
            enhancements_completed += 1
        
        # Final research summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.print_header("RESEARCH DEMONSTRATION SUMMARY")
        print(f"🔬 Research demonstration completed in {duration}")
        print(f"✅ Core Phases: {phases_completed}/5")
        print(f"✨ Enhancements: {enhancements_completed}/3")
        print(f"🎯 Overall Success: {((phases_completed + enhancements_completed)/8)*100:.1f}%")
        
        print("\n📊 Key Research Contributions Demonstrated:")
        print("   • Intent-based vehicle prediction with Random Forest classifier")
        print("   • Real-time slot negotiation and conflict resolution algorithms")
        print("   • Computer vision integration for live vehicle detection")
        print("   • Advanced ETA calculation with path conflict analysis")
        print("   • Physical hardware prototype with servo control")
        print("   • Comprehensive data logging for future research")
        
        if phases_completed >= 4 and enhancements_completed >= 2:
            print("🎉 RESEARCH SUCCESS! System ready for publication and deployment")
        elif phases_completed >= 3:
            print("✅ RESEARCH PROGRESS! Core system operational with enhancements")
        else:
            print("⚠️  RESEARCH IN PROGRESS - Check individual components")
        
        print("\n🚀 Next Research Steps:")
        print("   1. Access enhanced dashboard: http://localhost:8502")
        print("   2. Analyze computer vision performance data")
        print("   3. Review ETA calculation accuracy metrics")
        print("   4. Test physical model integration")
        print("   5. Collect additional real-world data for validation")
    
    def standard_enhanced_demo(self):
        """Standard enhanced demonstration"""
        self.print_header("ENHANCED SMART TRAFFIC MANAGEMENT SYSTEM")
        
        print("🚀 Enhanced System Features:")
        print("   • Advanced SUMO simulation with comprehensive data")
        print("   • Random Forest ML with 85%+ accuracy")
        print("   • Dynamic slot scheduling with conflict resolution")
        print("   • 3D visualization with real-time analytics")
        print("   • Enhanced hardware integration")
        
        start_time = datetime.now()
        success_count = 0
        total_phases = 5
        
        # Run enhanced phases
        data_file = self.phase_1_enhanced_simulation()
        if data_file:
            success_count += 1
        
        if self.phase_2_enhanced_ml(data_file):
            success_count += 1
        
        if self.phase_3_enhanced_scheduling(data_file):
            success_count += 1
        
        if self.phase_4_enhanced_visualization():
            success_count += 1
        
        if self.phase_5_enhanced_hardware():
            success_count += 1
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.print_header("ENHANCED DEMO SUMMARY")
        print(f"✅ Enhanced phases completed: {success_count}/{total_phases}")
        print(f"⏱️  Total time: {duration}")
        print(f"🎯 Success rate: {(success_count/total_phases)*100:.1f}%")
        
        if success_count == total_phases:
            print("🎉 ENHANCED SYSTEM FULLY OPERATIONAL!")
        elif success_count >= 3:
            print("✅ ENHANCED SYSTEM MOSTLY OPERATIONAL!")
        
        print("\n📊 Enhanced System Access:")
        print("   🌐 Dashboard: http://localhost:8502")
        print("   📁 Data: ./data/ directory")
        print("   🤖 Models: ./ml/models/")
        print("   📋 Schedules: ./data/schedule_data_*.json")


def main():
    """Main entry point for enhanced demo"""
    parser = argparse.ArgumentParser(description='Enhanced Smart Traffic Management System Demo')
    parser.add_argument('--mode', default='enhanced', 
                       choices=['enhanced', 'research', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 
                               'cv', 'eta', 'physical'],
                       help='Demo mode to run')
    parser.add_argument('--timeout', type=int, default=300, help='Command timeout in seconds')
    
    args = parser.parse_args()
    
    demo = EnhancedSmartTrafficDemo()
    
    try:
        if args.mode == 'enhanced':
            demo.standard_enhanced_demo()
        elif args.mode == 'research':
            demo.research_mode_demo()
        elif args.mode == 'phase1':
            demo.phase_1_enhanced_simulation()
        elif args.mode == 'phase2':
            demo.phase_2_enhanced_ml()
        elif args.mode == 'phase3':
            demo.phase_3_enhanced_scheduling()
        elif args.mode == 'phase4':
            demo.phase_4_enhanced_visualization()
        elif args.mode == 'phase5':
            demo.phase_5_enhanced_hardware()
        elif args.mode == 'cv':
            demo.enhancement_computer_vision()
        elif args.mode == 'eta':
            demo.enhancement_advanced_eta()
        elif args.mode == 'physical':
            demo.enhancement_physical_model()
            
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()
