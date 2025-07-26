#!/usr/bin/env python3
"""
Traffic Light Control System
Smart Traffic Management System - Phase 5

This script controls traffic lights using GPIO on Raspberry Pi
based on slot assignments from the central scheduling system.
"""

import time
import socket
import json
import threading
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List
import argparse

# Try to import RPi.GPIO, fallback to simulation mode if not available
try:
    import RPi.GPIO as GPIO
    SIMULATION_MODE = False
    print("Running in hardware mode with RPi.GPIO")
except ImportError:
    SIMULATION_MODE = True
    print("Running in simulation mode (RPi.GPIO not available)")


class LightState(Enum):
    """Traffic light states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    FLASHING_RED = "flashing_red"
    FLASHING_YELLOW = "flashing_yellow"
    OFF = "off"


@dataclass
class LightPhase:
    """Traffic light phase configuration"""
    direction: str
    state: LightState
    duration: float
    priority: int = 0  # 0=normal, 1=emergency, 2=preemption


class TrafficLight:
    """Individual traffic light controller"""
    
    def __init__(self, light_id: str, direction: str, red_pin: int, yellow_pin: int, green_pin: int):
        self.light_id = light_id
        self.direction = direction
        self.red_pin = red_pin
        self.yellow_pin = yellow_pin
        self.green_pin = green_pin
        self.current_state = LightState.RED
        self.flash_active = False
        self.flash_thread = None
        
        if not SIMULATION_MODE:
            # Setup GPIO pins
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.red_pin, GPIO.OUT)
            GPIO.setup(self.yellow_pin, GPIO.OUT)
            GPIO.setup(self.green_pin, GPIO.OUT)
            
            # Initialize to red state
            self.set_state(LightState.RED)
    
    def set_state(self, state: LightState):
        """Set the traffic light state"""
        # Stop any active flashing
        self._stop_flashing()
        
        self.current_state = state
        
        if SIMULATION_MODE:
            print(f"[SIM] Light {self.light_id} ({self.direction}): {state.value}")
            return
        
        # Turn off all lights first
        GPIO.output(self.red_pin, GPIO.LOW)
        GPIO.output(self.yellow_pin, GPIO.LOW)
        GPIO.output(self.green_pin, GPIO.LOW)
        
        # Set appropriate state
        if state == LightState.RED:
            GPIO.output(self.red_pin, GPIO.HIGH)
        elif state == LightState.YELLOW:
            GPIO.output(self.yellow_pin, GPIO.HIGH)
        elif state == LightState.GREEN:
            GPIO.output(self.green_pin, GPIO.HIGH)
        elif state == LightState.FLASHING_RED:
            self._start_flashing(self.red_pin, 1.0)
        elif state == LightState.FLASHING_YELLOW:
            self._start_flashing(self.yellow_pin, 0.5)
        # OFF state keeps all lights off
        
        print(f"Light {self.light_id} ({self.direction}): {state.value}")
    
    def _start_flashing(self, pin: int, interval: float):
        """Start flashing a specific light"""
        self.flash_active = True
        
        def flash_loop():
            while self.flash_active:
                if not SIMULATION_MODE:
                    GPIO.output(pin, GPIO.HIGH)
                time.sleep(interval / 2)
                if not SIMULATION_MODE:
                    GPIO.output(pin, GPIO.LOW)
                time.sleep(interval / 2)
        
        self.flash_thread = threading.Thread(target=flash_loop)
        self.flash_thread.daemon = True
        self.flash_thread.start()
    
    def _stop_flashing(self):
        """Stop any active flashing"""
        if self.flash_active:
            self.flash_active = False
            if self.flash_thread:
                self.flash_thread.join(timeout=1.0)


class IntersectionController:
    """Main intersection traffic light controller"""
    
    def __init__(self, intersection_id: str, server_host="localhost", server_port=8889):
        self.intersection_id = intersection_id
        self.server_host = server_host
        self.server_port = server_port
        
        self.lights: Dict[str, TrafficLight] = {}
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.running = False
        
        # Default timing parameters
        self.min_green_time = 10.0
        self.yellow_time = 3.0
        self.red_clearance_time = 1.0
        self.emergency_mode = False
        
        # Communication
        self.control_thread = None
        self.last_schedule_update = None
        
        # Standard 4-phase cycle for intersection
        self.standard_phases = [
            # Phase 0: North-South green
            {'ns_direction': LightState.GREEN, 'ew_direction': LightState.RED, 'duration': 30.0},
            # Phase 1: North-South yellow
            {'ns_direction': LightState.YELLOW, 'ew_direction': LightState.RED, 'duration': 3.0},
            # Phase 2: East-West green  
            {'ns_direction': LightState.RED, 'ew_direction': LightState.GREEN, 'duration': 30.0},
            # Phase 3: East-West yellow
            {'ns_direction': LightState.RED, 'ew_direction': LightState.YELLOW, 'duration': 3.0},
        ]
    
    def add_light(self, light_id: str, direction: str, red_pin: int, yellow_pin: int, green_pin: int):
        """Add a traffic light to the intersection"""
        light = TrafficLight(light_id, direction, red_pin, yellow_pin, green_pin)
        self.lights[direction] = light
        print(f"Added traffic light for {direction} (pins: R={red_pin}, Y={yellow_pin}, G={green_pin})")
    
    def start_control(self):
        """Start the traffic light control system"""
        self.running = True
        
        # Initialize all lights to red
        for light in self.lights.values():
            light.set_state(LightState.RED)
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("Traffic light control system started")
    
    def stop_control(self):
        """Stop the traffic light control system"""
        self.running = False
        
        if self.control_thread:
            self.control_thread.join()
        
        # Set all lights to flashing red (fail-safe)
        for light in self.lights.values():
            light.set_state(LightState.FLASHING_RED)
        
        time.sleep(2)  # Let flashing start
        
        if not SIMULATION_MODE:
            GPIO.cleanup()
        
        print("Traffic light control system stopped")
    
    def _control_loop(self):
        """Main control loop"""
        print("Traffic control loop started")
        
        while self.running:
            try:
                # Check for schedule updates from central system
                schedule_update = self._check_for_schedule_update()
                
                if schedule_update:
                    self._apply_schedule_update(schedule_update)
                else:
                    # Run standard timing if no schedule available
                    self._run_standard_timing()
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                self._emergency_mode()
                time.sleep(5)
    
    def _check_for_schedule_update(self) -> Optional[Dict]:
        """Check for schedule updates from central server"""
        try:
            # Create UDP socket to receive schedule updates
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.1)  # Non-blocking
                sock.bind(('', self.server_port))
                
                data, addr = sock.recvfrom(1024)
                schedule = json.loads(data.decode('utf-8'))
                
                self.last_schedule_update = datetime.now()
                return schedule
                
        except (socket.timeout, socket.error):
            # No data available, which is normal
            pass
        except Exception as e:
            print(f"Error receiving schedule update: {e}")
        
        return None
    
    def _apply_schedule_update(self, schedule: Dict):
        """Apply schedule update from central system"""
        try:
            # Extract timing information
            current_slot = schedule.get('current_slot', {})
            movement_type = current_slot.get('movement_type')
            slot_duration = current_slot.get('duration', 30.0)
            emergency_vehicles = schedule.get('emergency_vehicles', [])
            
            # Handle emergency vehicles first
            if emergency_vehicles:
                self._handle_emergency_vehicles(emergency_vehicles)
                return
            
            # Apply normal schedule based on movement type
            if movement_type == 'straight':
                # Prioritize main directions (NS and EW)
                self._set_phase_for_straight_movement()
            elif movement_type == 'left':
                # Handle left turn phases
                self._set_phase_for_left_turns()
            elif movement_type == 'right':
                # Right turns usually don't need special timing
                self._set_phase_for_right_turns()
            else:
                # Default to standard timing
                self._run_standard_timing()
            
            print(f"Applied schedule update: {movement_type}, duration: {slot_duration}s")
            
        except Exception as e:
            print(f"Error applying schedule update: {e}")
            self._run_standard_timing()
    
    def _handle_emergency_vehicles(self, emergency_vehicles: List[Dict]):
        """Handle emergency vehicle preemption"""
        print(f"Emergency preemption activated for {len(emergency_vehicles)} vehicles")
        
        self.emergency_mode = True
        
        # Determine which direction emergency vehicles are coming from
        emergency_directions = set()
        for vehicle in emergency_vehicles:
            lane = vehicle.get('lane', '')
            if 'north' in lane:
                emergency_directions.add('north')
            elif 'south' in lane:
                emergency_directions.add('south')
            elif 'east' in lane:
                emergency_directions.add('east')
            elif 'west' in lane:
                emergency_directions.add('west')
        
        # Clear path for emergency vehicles
        if 'north' in emergency_directions or 'south' in emergency_directions:
            # Give green to north-south
            self._set_lights_state('north', LightState.GREEN)
            self._set_lights_state('south', LightState.GREEN)
            self._set_lights_state('east', LightState.RED)
            self._set_lights_state('west', LightState.RED)
        elif 'east' in emergency_directions or 'west' in emergency_directions:
            # Give green to east-west
            self._set_lights_state('north', LightState.RED)
            self._set_lights_state('south', LightState.RED)
            self._set_lights_state('east', LightState.GREEN)
            self._set_lights_state('west', LightState.GREEN)
    
    def _set_phase_for_straight_movement(self):
        """Set optimal phase for straight movements"""
        # Alternate between NS and EW based on current phase
        current_time = time.time()
        phase_elapsed = current_time - self.phase_start_time
        
        if self.current_phase in [0, 1]:  # NS phases
            if phase_elapsed >= self.min_green_time:
                self._transition_to_next_phase()
        else:  # EW phases
            if phase_elapsed >= self.min_green_time:
                self._transition_to_next_phase()
    
    def _set_phase_for_left_turns(self):
        """Set phase for protected left turns"""
        # Could implement protected left turn phases here
        # For now, use standard timing
        self._run_standard_timing()
    
    def _set_phase_for_right_turns(self):
        """Set phase for right turns (usually concurrent with through movements)"""
        self._run_standard_timing()
    
    def _run_standard_timing(self):
        """Run standard intersection timing"""
        current_time = time.time()
        phase_elapsed = current_time - self.phase_start_time
        
        current_phase_config = self.standard_phases[self.current_phase]
        phase_duration = current_phase_config['duration']
        
        if phase_elapsed >= phase_duration:
            self._transition_to_next_phase()
        else:
            # Ensure lights are in correct state for current phase
            ns_state = current_phase_config['ns_direction']
            ew_state = current_phase_config['ew_direction']
            
            self._set_lights_state('north', ns_state)
            self._set_lights_state('south', ns_state)
            self._set_lights_state('east', ew_state)
            self._set_lights_state('west', ew_state)
    
    def _transition_to_next_phase(self):
        """Transition to the next phase in the cycle"""
        self.current_phase = (self.current_phase + 1) % len(self.standard_phases)
        self.phase_start_time = time.time()
        self.emergency_mode = False
        
        print(f"Transitioning to phase {self.current_phase}")
    
    def _set_lights_state(self, direction: str, state: LightState):
        """Set state for lights in a specific direction"""
        if direction in self.lights:
            self.lights[direction].set_state(state)
    
    def _emergency_mode(self):
        """Enter emergency mode (flashing red)"""
        print("Entering emergency mode - all lights flashing red")
        for light in self.lights.values():
            light.set_state(LightState.FLASHING_RED)
        
        time.sleep(10)  # Emergency mode for 10 seconds
    
    def get_status(self) -> Dict:
        """Get current intersection status"""
        return {
            'intersection_id': self.intersection_id,
            'running': self.running,
            'current_phase': self.current_phase,
            'emergency_mode': self.emergency_mode,
            'lights': {
                direction: light.current_state.value
                for direction, light in self.lights.items()
            },
            'last_schedule_update': self.last_schedule_update.isoformat() if self.last_schedule_update else None
        }


def setup_intersection_lights(controller: IntersectionController):
    """Setup traffic lights for a 4-way intersection"""
    
    # Light configuration for intersection
    # Pin numbers are for Raspberry Pi GPIO (BCM numbering)
    light_configs = [
        # Direction, Red Pin, Yellow Pin, Green Pin
        ('north', 2, 3, 4),      # North approach
        ('south', 14, 15, 18),   # South approach  
        ('east', 7, 8, 9),       # East approach
        ('west', 10, 11, 25),    # West approach
    ]
    
    for direction, red_pin, yellow_pin, green_pin in light_configs:
        controller.add_light(f"light_{direction}", direction, red_pin, yellow_pin, green_pin)


def main():
    """Main function to run traffic light control system"""
    parser = argparse.ArgumentParser(description='Traffic Light Control System')
    parser.add_argument('--intersection-id', default='intersection_001', help='Intersection identifier')
    parser.add_argument('--server-host', default='localhost', help='Central server hostname')
    parser.add_argument('--server-port', type=int, default=8889, help='Central server port')
    parser.add_argument('--simulation', action='store_true', help='Force simulation mode')
    parser.add_argument('--duration', type=int, default=0, help='Run duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    global SIMULATION_MODE
    if args.simulation:
        SIMULATION_MODE = True
        print("Forced simulation mode enabled")
    
    print("=" * 60)
    print("TRAFFIC LIGHT CONTROL SYSTEM")
    print("=" * 60)
    print(f"Intersection: {args.intersection_id}")
    print(f"Mode: {'Simulation' if SIMULATION_MODE else 'Hardware'}")
    print(f"Server: {args.server_host}:{args.server_port}")
    
    # Create intersection controller
    controller = IntersectionController(args.intersection_id, args.server_host, args.server_port)
    
    # Setup traffic lights
    setup_intersection_lights(controller)
    
    # Start control system
    controller.start_control()
    
    try:
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely... Press Ctrl+C to stop")
            while True:
                # Show status every 15 seconds
                time.sleep(15)
                status = controller.get_status()
                lights_status = ", ".join([f"{d}: {s}" for d, s in status['lights'].items()])
                print(f"Phase {status['current_phase']}: {lights_status}")
    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    
    finally:
        controller.stop_control()
        print("Traffic light control system stopped")


if __name__ == "__main__":
    main()
