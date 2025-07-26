#!/usr/bin/env python3
"""
Traffic Light Controller - Hardware Simulation
Smart Traffic Management System - Phase 5

Simulates Raspberry Pi GPIO traffic light control for demonstration.
"""

import time
import argparse
from datetime import datetime
from enum import Enum


class LightState(Enum):
    """Traffic light states"""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class TrafficLightSimulator:
    """Simulates GPIO-controlled traffic lights"""
    
    def __init__(self, intersection_name="Main Intersection"):
        self.intersection_name = intersection_name
        self.directions = ["North", "South", "East", "West"]
        self.current_states = {direction: LightState.RED for direction in self.directions}
        self.cycle_duration = 30  # seconds per direction
        self.yellow_duration = 3  # seconds for yellow light
        
    def display_status(self):
        """Display current traffic light status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] {self.intersection_name} - Traffic Light Status:")
        print("=" * 60)
        
        for direction in self.directions:
            state = self.current_states[direction]
            if state == LightState.RED:
                color_symbol = "RED"
            elif state == LightState.YELLOW:
                color_symbol = "YEL"
            else:  # GREEN
                color_symbol = "GRN"
            
            print(f"  {direction:6s}: [{color_symbol}] {state.value}")
        
        print("=" * 60)
    
    def set_light(self, direction, state):
        """Set a specific direction's light state"""
        if direction in self.directions:
            self.current_states[direction] = state
            print(f">> {direction} light changed to {state.value}")
    
    def set_direction_green(self, direction):
        """Set one direction to green, others to red"""
        for dir_name in self.directions:
            if dir_name == direction:
                self.set_light(dir_name, LightState.GREEN)
            else:
                self.set_light(dir_name, LightState.RED)
    
    def transition_yellow(self, current_direction):
        """Transition current green direction to yellow"""
        self.set_light(current_direction, LightState.YELLOW)
        print(f">> {current_direction} entering yellow phase...")
        time.sleep(self.yellow_duration)
    
    def run_standard_cycle(self, duration_minutes=1):
        """Run a standard traffic light cycle"""
        print(f">> Starting {duration_minutes}-minute traffic light cycle...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        direction_index = 0
        
        while time.time() < end_time:
            current_direction = self.directions[direction_index]
            
            # Set current direction to green
            print(f"\n>> Activating {current_direction} direction...")
            self.set_direction_green(current_direction)
            self.display_status()
            
            # Wait for green phase
            green_time = min(self.cycle_duration, end_time - time.time())
            if green_time <= self.yellow_duration:
                break
                
            print(f"   Green phase: {green_time:.1f} seconds")
            time.sleep(green_time - self.yellow_duration)
            
            # Yellow transition
            if time.time() < end_time - self.yellow_duration:
                self.transition_yellow(current_direction)
            
            # Move to next direction
            direction_index = (direction_index + 1) % len(self.directions)
        
        # End cycle - all red
        for direction in self.directions:
            self.set_light(direction, LightState.RED)
        
        print(f"\n>> Cycle complete - All directions RED")
        self.display_status()
    
    def run_smart_control(self, duration_minutes=1):
        """Simulate smart traffic control based on vehicle density"""
        print(f">> Starting {duration_minutes}-minute SMART traffic control...")
        
        # Simulate varying traffic densities
        traffic_patterns = [
            {"North": 0.8, "South": 0.6, "East": 0.9, "West": 0.4},
            {"North": 0.3, "South": 0.7, "East": 0.5, "West": 0.8},
            {"North": 0.6, "South": 0.4, "East": 0.3, "West": 0.7},
        ]
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        pattern_index = 0
        
        while time.time() < end_time:
            pattern = traffic_patterns[pattern_index % len(traffic_patterns)]
            
            # Find direction with highest traffic
            max_traffic_dir = max(pattern.keys(), key=lambda k: pattern[k])
            max_traffic = pattern[max_traffic_dir]
            
            print(f"\n>> Traffic Analysis:")
            for direction, density in pattern.items():
                bar = "|" * int(density * 10)
                print(f"   {direction:6s}: {density:.1f} {bar}")
            
            print(f"\n>> Prioritizing {max_traffic_dir} (density: {max_traffic:.1f})")
            
            # Calculate dynamic timing based on traffic
            green_duration = 10 + (max_traffic * 20)  # 10-30 seconds
            green_duration = min(green_duration, end_time - time.time())
            
            if green_duration <= self.yellow_duration:
                break
            
            # Activate highest priority direction
            self.set_direction_green(max_traffic_dir)
            self.display_status()
            
            print(f"   Smart green phase: {green_duration:.1f} seconds")
            time.sleep(green_duration - self.yellow_duration)
            
            # Yellow transition
            if time.time() < end_time - self.yellow_duration:
                self.transition_yellow(max_traffic_dir)
            
            pattern_index += 1
        
        # End smart control
        for direction in self.directions:
            self.set_light(direction, LightState.RED)
        
        print(f"\n>> Smart control complete - All directions RED")
        self.display_status()
    
    def emergency_override(self, emergency_direction="North", duration=10):
        """Emergency vehicle override"""
        print(f"\n>> EMERGENCY OVERRIDE - {emergency_direction} direction!")
        
        # Immediate yellow for all active greens
        for direction in self.directions:
            if self.current_states[direction] == LightState.GREEN:
                self.set_light(direction, LightState.YELLOW)
        
        time.sleep(self.yellow_duration)
        
        # Clear path for emergency vehicle
        self.set_direction_green(emergency_direction)
        print(f">> Emergency path cleared for {emergency_direction}")
        self.display_status()
        
        time.sleep(duration)
        
        # Return to normal operation
        self.set_light(emergency_direction, LightState.YELLOW)
        time.sleep(self.yellow_duration)
        self.set_light(emergency_direction, LightState.RED)
        
        print(">> Emergency override complete")


def demo_traffic_lights(duration=30):
    """Run traffic light demonstration"""
    print("TRAFFIC LIGHT CONTROLLER DEMO")
    print("=" * 50)
    print("Simulating Raspberry Pi GPIO traffic light control")
    print("=" * 50)
    
    # Initialize traffic light controller
    controller = TrafficLightSimulator("Smart Intersection Demo")
    
    # Initial status
    print("\n>> Initial State:")
    controller.display_status()
    time.sleep(2)
    
    # Run standard cycle for 40% of duration
    standard_duration = duration * 0.4 / 60  # Convert to minutes
    controller.run_standard_cycle(standard_duration)
    time.sleep(1)
    
    # Emergency override demo
    print(f"\n>> Emergency Override Demo...")
    controller.emergency_override("East", 5)
    time.sleep(1)
    
    # Run smart control for remaining time
    smart_duration = duration * 0.5 / 60  # Convert to minutes
    controller.run_smart_control(smart_duration)
    
    print(f"\n>> Traffic light demo completed!")
    print(f">> Total duration: {duration} seconds")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Traffic Light Controller Simulation')
    parser.add_argument('--demo', action='store_true', help='Run demonstration mode')
    parser.add_argument('--duration', type=int, default=60, help='Demo duration in seconds')
    parser.add_argument('--mode', default='standard', choices=['standard', 'smart', 'emergency'],
                       help='Control mode')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_traffic_lights(args.duration)
    else:
        controller = TrafficLightSimulator()
        
        if args.mode == 'standard':
            controller.run_standard_cycle(1)
        elif args.mode == 'smart':
            controller.run_smart_control(1)
        elif args.mode == 'emergency':
            controller.emergency_override("North", 10)


if __name__ == "__main__":
    main()
