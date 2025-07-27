#!/usr/bin/env python3
"""
Physical Model Control System
Smart Traffic Management System - Enhanced Hardware Integration

This module controls physical components including servo motors for gates,
LEDs for traffic signals, and sensors for vehicle detection in a miniature
traffic intersection model.

Features:
- Servo motor control for intersection gates
- LED traffic light control
- Physical model synchronization with simulation
- Emergency protocols with physical hardware
"""

import time
import threading
import json
import socket
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import argparse

# Try to import RPi.GPIO and servo control
try:
    import RPi.GPIO as GPIO
    from gpiozero import Servo, LED, Button
    HARDWARE_AVAILABLE = True
    print("Hardware mode: RPi.GPIO available")
except (ImportError, RuntimeError):
    HARDWARE_AVAILABLE = False
    GPIO = None
    Servo = LED = Button = None
    print("Simulation mode: Hardware libraries not available")


class GateState(Enum):
    """Gate position states"""
    CLOSED = "closed"      # Gate blocking traffic (0 degrees)
    OPEN = "open"         # Gate allowing traffic (90 degrees)
    OPENING = "opening"   # Gate in motion to open
    CLOSING = "closing"   # Gate in motion to close


class SignalState(Enum):
    """Traffic signal states"""
    RED = "red"
    YELLOW = "yellow" 
    GREEN = "green"
    FLASHING_RED = "flashing_red"
    FLASHING_YELLOW = "flashing_yellow"


@dataclass
class GateConfig:
    """Configuration for a servo-controlled gate"""
    gate_id: str
    gpio_pin: int
    lane: str
    open_angle: float = 90.0   # degrees
    closed_angle: float = 0.0  # degrees
    movement_speed: float = 1.0  # seconds for full movement


@dataclass
class SignalConfig:
    """Configuration for LED traffic signals"""
    signal_id: str
    red_pin: int
    yellow_pin: int
    green_pin: int
    direction: str  # north, south, east, west


class ServoGate:
    """Individual servo-controlled gate"""
    
    def __init__(self, config: GateConfig):
        self.config = config
        self.current_state = GateState.CLOSED
        self.target_state = GateState.CLOSED
        self.servo = None
        self.movement_thread = None
        self.moving = False
        
        if HARDWARE_AVAILABLE:
            self.servo = Servo(config.gpio_pin)
            self._set_physical_position(config.closed_angle)
        
        print(f"Gate {config.gate_id} initialized on pin {config.gpio_pin}")
    
    def _set_physical_position(self, angle_degrees: float):
        """Set physical servo position"""
        if not HARDWARE_AVAILABLE or not self.servo:
            return
        
        # Convert degrees to servo value (-1 to 1)
        # Assuming 0 degrees = -1, 90 degrees = 0, 180 degrees = 1
        servo_value = (angle_degrees - 90) / 90.0
        servo_value = max(-1.0, min(1.0, servo_value))
        
        try:
            self.servo.value = servo_value
        except Exception as e:
            print(f"Servo control error: {e}")
    
    def open_gate(self, duration: float = None):
        """Open the gate"""
        if self.current_state == GateState.OPEN:
            return
        
        self.target_state = GateState.OPEN
        self.current_state = GateState.OPENING
        
        if duration:
            # Schedule automatic closure
            def auto_close():
                time.sleep(duration)
                self.close_gate()
            
            auto_close_thread = threading.Thread(target=auto_close)
            auto_close_thread.daemon = True
            auto_close_thread.start()
        
        self._move_to_position(self.config.open_angle)
    
    def close_gate(self):
        """Close the gate"""
        if self.current_state == GateState.CLOSED:
            return
        
        self.target_state = GateState.CLOSED
        self.current_state = GateState.CLOSING
        self._move_to_position(self.config.closed_angle)
    
    def _move_to_position(self, target_angle: float):
        """Move servo to target position with smooth motion"""
        if self.movement_thread and self.movement_thread.is_alive():
            return  # Already moving
        
        self.movement_thread = threading.Thread(
            target=self._smooth_movement, 
            args=(target_angle,)
        )
        self.movement_thread.daemon = True
        self.movement_thread.start()
    
    def _smooth_movement(self, target_angle: float):
        """Perform smooth servo movement"""
        self.moving = True
        
        if HARDWARE_AVAILABLE and self.servo:
            current_angle = self.config.closed_angle if self.current_state == GateState.CLOSED else self.config.open_angle
            steps = 20  # Number of movement steps
            angle_step = (target_angle - current_angle) / steps
            time_step = self.config.movement_speed / steps
            
            for i in range(steps + 1):
                angle = current_angle + (angle_step * i)
                self._set_physical_position(angle)
                time.sleep(time_step)
        else:
            # Simulation mode
            time.sleep(self.config.movement_speed)
        
        # Update state
        if target_angle == self.config.open_angle:
            self.current_state = GateState.OPEN
        else:
            self.current_state = GateState.CLOSED
        
        self.moving = False
        print(f"Gate {self.config.gate_id} moved to {self.current_state.value}")
    
    def get_status(self) -> Dict:
        """Get current gate status"""
        return {
            'gate_id': self.config.gate_id,
            'lane': self.config.lane,
            'state': self.current_state.value,
            'target_state': self.target_state.value,
            'moving': self.moving,
            'gpio_pin': self.config.gpio_pin
        }


class TrafficSignal:
    """LED-based traffic signal control"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.current_state = SignalState.RED
        self.leds = {}
        self.flashing_thread = None
        self.flashing_active = False
        
        if HARDWARE_AVAILABLE:
            self.leds['red'] = LED(config.red_pin)
            self.leds['yellow'] = LED(config.yellow_pin)
            self.leds['green'] = LED(config.green_pin)
            
            # Initialize to red
            self._set_physical_state(SignalState.RED)
        
        print(f"Traffic signal {config.signal_id} initialized for {config.direction}")
    
    def _set_physical_state(self, state: SignalState):
        """Set physical LED state"""
        if not HARDWARE_AVAILABLE:
            return
        
        # Turn off all LEDs first
        for led in self.leds.values():
            led.off()
        
        # Set appropriate LED
        if state == SignalState.RED:
            self.leds['red'].on()
        elif state == SignalState.YELLOW:
            self.leds['yellow'].on()
        elif state == SignalState.GREEN:
            self.leds['green'].on()
    
    def set_signal(self, state: SignalState):
        """Set traffic signal state"""
        self._stop_flashing()
        
        self.current_state = state
        
        if state in [SignalState.FLASHING_RED, SignalState.FLASHING_YELLOW]:
            self._start_flashing(state)
        else:
            self._set_physical_state(state)
        
        print(f"Signal {self.config.signal_id} set to {state.value}")
    
    def _start_flashing(self, flash_state: SignalState):
        """Start flashing mode"""
        self.flashing_active = True
        self.flashing_thread = threading.Thread(
            target=self._flash_loop, 
            args=(flash_state,)
        )
        self.flashing_thread.daemon = True
        self.flashing_thread.start()
    
    def _stop_flashing(self):
        """Stop flashing mode"""
        self.flashing_active = False
        if self.flashing_thread:
            self.flashing_thread.join(timeout=1)
    
    def _flash_loop(self, flash_state: SignalState):
        """Flashing loop for emergency signals"""
        led_color = 'red' if flash_state == SignalState.FLASHING_RED else 'yellow'
        
        while self.flashing_active:
            if HARDWARE_AVAILABLE and led_color in self.leds:
                self.leds[led_color].on()
                time.sleep(0.5)
                self.leds[led_color].off()
                time.sleep(0.5)
            else:
                time.sleep(1)  # Simulation delay
    
    def get_status(self) -> Dict:
        """Get current signal status"""
        return {
            'signal_id': self.config.signal_id,
            'direction': self.config.direction,
            'state': self.current_state.value,
            'flashing': self.flashing_active,
            'pins': {
                'red': self.config.red_pin,
                'yellow': self.config.yellow_pin,
                'green': self.config.green_pin
            }
        }


class PhysicalIntersectionController:
    """Main controller for physical intersection model"""
    
    def __init__(self):
        self.gates = {}
        self.signals = {}
        self.emergency_mode = False
        self.running = False
        
        # Communication
        self.server_socket = None
        self.command_queue = []
        self.command_lock = threading.Lock()
        
        # Initialize hardware components
        self._initialize_gates()
        self._initialize_signals()
        
        print("Physical intersection controller initialized")
    
    def _initialize_gates(self):
        """Initialize all servo gates"""
        gate_configs = [
            GateConfig("north_gate", 18, "north_approach", 90, 0, 1.0),
            GateConfig("south_gate", 19, "south_approach", 90, 0, 1.0),
            GateConfig("east_gate", 20, "east_approach", 90, 0, 1.0),
            GateConfig("west_gate", 21, "west_approach", 90, 0, 1.0),
        ]
        
        for config in gate_configs:
            self.gates[config.gate_id] = ServoGate(config)
    
    def _initialize_signals(self):
        """Initialize all traffic signals"""
        signal_configs = [
            SignalConfig("north_signal", 2, 3, 4, "north"),
            SignalConfig("south_signal", 14, 15, 18, "south"),
            SignalConfig("east_signal", 7, 8, 9, "east"),
            SignalConfig("west_signal", 10, 11, 25, "west"),
        ]
        
        for config in signal_configs:
            self.signals[config.signal_id] = TrafficSignal(config)
    
    def control_gate(self, gate_id: str, action: str, duration: float = None):
        """Control a specific gate"""
        if gate_id not in self.gates:
            print(f"Unknown gate ID: {gate_id}")
            return False
        
        gate = self.gates[gate_id]
        
        if action == "open":
            gate.open_gate(duration)
        elif action == "close":
            gate.close_gate()
        else:
            print(f"Unknown gate action: {action}")
            return False
        
        return True
    
    def control_signal(self, signal_id: str, state: str):
        """Control a specific traffic signal"""
        if signal_id not in self.signals:
            print(f"Unknown signal ID: {signal_id}")
            return False
        
        try:
            signal_state = SignalState(state.lower())
            self.signals[signal_id].set_signal(signal_state)
            return True
        except ValueError:
            print(f"Invalid signal state: {state}")
            return False
    
    def execute_traffic_plan(self, plan: Dict):
        """Execute a comprehensive traffic plan"""
        print(f"Executing traffic plan: {plan.get('plan_id', 'unknown')}")
        
        # Execute gate commands
        for gate_command in plan.get('gate_commands', []):
            gate_id = gate_command.get('gate_id')
            action = gate_command.get('action')
            duration = gate_command.get('duration')
            delay = gate_command.get('delay', 0)
            
            if delay > 0:
                time.sleep(delay)
            
            self.control_gate(gate_id, action, duration)
        
        # Execute signal commands
        for signal_command in plan.get('signal_commands', []):
            signal_id = signal_command.get('signal_id')
            state = signal_command.get('state')
            delay = signal_command.get('delay', 0)
            
            if delay > 0:
                time.sleep(delay)
            
            self.control_signal(signal_id, state)
    
    def emergency_protocol(self, emergency_direction: str):
        """Execute emergency vehicle protocol"""
        print(f"EMERGENCY PROTOCOL: Clearing path for {emergency_direction}")
        
        self.emergency_mode = True
        
        # Set all signals to red
        for signal in self.signals.values():
            signal.set_signal(SignalState.RED)
        
        time.sleep(1)  # Brief pause
        
        # Open gate for emergency direction and set signal to green
        emergency_gate_id = f"{emergency_direction}_gate"
        emergency_signal_id = f"{emergency_direction}_signal"
        
        if emergency_gate_id in self.gates:
            self.control_gate(emergency_gate_id, "open", 10.0)  # Open for 10 seconds
        
        if emergency_signal_id in self.signals:
            self.control_signal(emergency_signal_id, "green")
        
        # Flash other signals
        for signal_id, signal in self.signals.items():
            if signal_id != emergency_signal_id:
                signal.set_signal(SignalState.FLASHING_RED)
        
        # Schedule return to normal operation
        def return_to_normal():
            time.sleep(10)  # Emergency duration
            self.emergency_mode = False
            self.reset_intersection()
        
        emergency_thread = threading.Thread(target=return_to_normal)
        emergency_thread.daemon = True
        emergency_thread.start()
    
    def reset_intersection(self):
        """Reset intersection to safe state"""
        print("Resetting intersection to safe state")
        
        # Close all gates
        for gate in self.gates.values():
            gate.close_gate()
        
        # Set all signals to red
        for signal in self.signals.values():
            signal.set_signal(SignalState.RED)
        
        self.emergency_mode = False
    
    def start_command_server(self, port: int = 8890):
        """Start command server for remote control"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(('localhost', port))
        self.running = True
        
        command_thread = threading.Thread(target=self._command_loop)
        command_thread.daemon = True
        command_thread.start()
        
        print(f"Command server started on port {port}")
    
    def _command_loop(self):
        """Main command processing loop"""
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                command = json.loads(data.decode('utf-8'))
                
                with self.command_lock:
                    self.command_queue.append(command)
                
                self._process_command(command)
                
            except Exception as e:
                print(f"Command processing error: {e}")
    
    def _process_command(self, command: Dict):
        """Process individual command"""
        command_type = command.get('type')
        
        if command_type == 'gate_control':
            gate_id = command.get('gate_id')
            action = command.get('action')
            duration = command.get('duration')
            self.control_gate(gate_id, action, duration)
            
        elif command_type == 'signal_control':
            signal_id = command.get('signal_id')
            state = command.get('state')
            self.control_signal(signal_id, state)
            
        elif command_type == 'traffic_plan':
            self.execute_traffic_plan(command.get('plan', {}))
            
        elif command_type == 'emergency':
            direction = command.get('direction', 'north')
            self.emergency_protocol(direction)
            
        elif command_type == 'reset':
            self.reset_intersection()
            
        elif command_type == 'status':
            self._send_status_response()
        
        else:
            print(f"Unknown command type: {command_type}")
    
    def _send_status_response(self):
        """Send status response to clients"""
        status = self.get_system_status()
        # In a full implementation, this would send status back to clients
        print("Status requested:", json.dumps(status, indent=2))
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': self.emergency_mode,
            'hardware_available': HARDWARE_AVAILABLE,
            'gates': {gate_id: gate.get_status() for gate_id, gate in self.gates.items()},
            'signals': {signal_id: signal.get_status() for signal_id, signal in self.signals.items()},
            'command_queue_size': len(self.command_queue)
        }
    
    def demo_sequence(self):
        """Run a demonstration sequence"""
        print("Starting physical model demonstration...")
        
        # Reset to start
        self.reset_intersection()
        time.sleep(2)
        
        # Demo normal traffic flow
        print("Demo: Normal traffic flow")
        
        # North-South green
        self.control_signal("north_signal", "green")
        self.control_signal("south_signal", "green")
        self.control_gate("north_gate", "open", 5)
        self.control_gate("south_gate", "open", 5)
        time.sleep(6)
        
        # Change to East-West
        self.control_signal("north_signal", "yellow")
        self.control_signal("south_signal", "yellow")
        time.sleep(2)
        
        self.control_signal("north_signal", "red")
        self.control_signal("south_signal", "red")
        self.control_signal("east_signal", "green")
        self.control_signal("west_signal", "green")
        self.control_gate("east_gate", "open", 5)
        self.control_gate("west_gate", "open", 5)
        time.sleep(6)
        
        # Demo emergency protocol
        print("Demo: Emergency protocol")
        self.emergency_protocol("north")
        time.sleep(12)
        
        print("Physical model demonstration completed")
    
    def stop_system(self):
        """Stop the physical control system"""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        # Reset all hardware to safe state
        self.reset_intersection()
        
        if HARDWARE_AVAILABLE:
            GPIO.cleanup()
        
        print("Physical control system stopped")


def main():
    """Main function for physical model control"""
    parser = argparse.ArgumentParser(description='Physical Intersection Model Controller')
    parser.add_argument('--port', type=int, default=8890, help='Command server port')
    parser.add_argument('--demo', action='store_true', help='Run demonstration sequence')
    parser.add_argument('--duration', type=int, default=0, help='Run duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHYSICAL INTERSECTION MODEL CONTROLLER")
    print("=" * 60)
    print(f"Hardware Mode: {'Enabled' if HARDWARE_AVAILABLE else 'Simulation'}")
    
    # Create controller
    controller = PhysicalIntersectionController()
    
    try:
        # Start command server
        controller.start_command_server(args.port)
        
        if args.demo:
            # Run demonstration
            time.sleep(2)  # Allow server to start
            controller.demo_sequence()
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely... Press Ctrl+C to stop")
            while True:
                time.sleep(10)
                status = controller.get_system_status()
                print(f"System status: {len(status['gates'])} gates, {len(status['signals'])} signals")
    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    
    finally:
        controller.stop_system()


if __name__ == "__main__":
    main()
