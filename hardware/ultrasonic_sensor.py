#!/usr/bin/env python3
"""
Ultrasonic Sensor Vehicle Detection
Smart Traffic Management System - Phase 5

This script interfaces with HC-SR04 ultrasonic sensors on Raspberry Pi
to detect vehicle presence and communicate with the central scheduling system.
"""

import time
import socket
import json
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List
import argparse

# Try to import RPi.GPIO, fallback to simulation mode if not available
try:
    import RPi.GPIO as GPIO
    SIMULATION_MODE = False
    print("Running in hardware mode with RPi.GPIO")
except ImportError:
    import random
    SIMULATION_MODE = True
    print("Running in simulation mode (RPi.GPIO not available)")


@dataclass
class VehicleDetection:
    """Represents a vehicle detection event"""
    sensor_id: str
    lane: str
    distance: float
    timestamp: datetime
    confidence: float = 1.0


class UltrasonicSensor:
    """HC-SR04 Ultrasonic Sensor interface"""
    
    def __init__(self, sensor_id: str, trig_pin: int, echo_pin: int, lane: str):
        self.sensor_id = sensor_id
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.lane = lane
        self.max_distance = 400.0  # Maximum detection range in cm
        self.vehicle_threshold = 200.0  # Distance below which vehicle is detected
        
        if not SIMULATION_MODE:
            # Setup GPIO pins
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, False)
    
    def measure_distance(self) -> Optional[float]:
        """Measure distance using ultrasonic sensor"""
        if SIMULATION_MODE:
            # Simulate distance measurement
            # 70% chance of no vehicle (>200cm), 30% chance of vehicle detection
            if random.random() < 0.7:
                return random.uniform(250, 400)  # No vehicle
            else:
                return random.uniform(50, 180)   # Vehicle detected
        
        try:
            # Send trigger pulse
            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trig_pin, False)
            
            # Wait for echo start
            start_time = time.time()
            timeout_start = start_time
            while GPIO.input(self.echo_pin) == 0:
                start_time = time.time()
                if start_time - timeout_start > 0.1:  # 100ms timeout
                    return None
            
            # Wait for echo end
            stop_time = time.time()
            timeout_start = stop_time
            while GPIO.input(self.echo_pin) == 1:
                stop_time = time.time()
                if stop_time - timeout_start > 0.1:  # 100ms timeout
                    return None
            
            # Calculate distance
            time_elapsed = stop_time - start_time
            distance = (time_elapsed * 34300) / 2  # Speed of sound = 343 m/s
            
            return distance if distance <= self.max_distance else None
            
        except Exception as e:
            print(f"Error measuring distance on sensor {self.sensor_id}: {e}")
            return None
    
    def detect_vehicle(self) -> Optional[VehicleDetection]:
        """Detect if a vehicle is present"""
        distance = self.measure_distance()
        
        if distance is not None and distance < self.vehicle_threshold:
            # Vehicle detected
            confidence = max(0.5, min(1.0, (self.vehicle_threshold - distance) / self.vehicle_threshold))
            
            return VehicleDetection(
                sensor_id=self.sensor_id,
                lane=self.lane,
                distance=distance,
                timestamp=datetime.now(),
                confidence=confidence
            )
        
        return None


class VehicleDetectionSystem:
    """Main vehicle detection system managing multiple sensors"""
    
    def __init__(self, server_host="localhost", server_port=8888):
        self.sensors: Dict[str, UltrasonicSensor] = {}
        self.server_host = server_host
        self.server_port = server_port
        self.running = False
        self.detection_thread = None
        self.communication_thread = None
        self.detection_queue: List[VehicleDetection] = []
        self.detection_lock = threading.Lock()
        
        # Detection parameters
        self.detection_interval = 0.1  # Check sensors every 100ms
        self.send_interval = 1.0       # Send data every second
        
    def add_sensor(self, sensor_id: str, trig_pin: int, echo_pin: int, lane: str):
        """Add an ultrasonic sensor to the system"""
        sensor = UltrasonicSensor(sensor_id, trig_pin, echo_pin, lane)
        self.sensors[sensor_id] = sensor
        print(f"Added sensor {sensor_id} for lane {lane} (pins: trig={trig_pin}, echo={echo_pin})")
    
    def start_detection(self):
        """Start the vehicle detection system"""
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start communication thread
        self.communication_thread = threading.Thread(target=self._communication_loop)
        self.communication_thread.daemon = True
        self.communication_thread.start()
        
        print("Vehicle detection system started")
    
    def stop_detection(self):
        """Stop the vehicle detection system"""
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join()
        
        if self.communication_thread:
            self.communication_thread.join()
        
        if not SIMULATION_MODE:
            GPIO.cleanup()
        
        print("Vehicle detection system stopped")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        print("Detection loop started")
        
        while self.running:
            try:
                # Check all sensors
                for sensor_id, sensor in self.sensors.items():
                    detection = sensor.detect_vehicle()
                    
                    if detection:
                        with self.detection_lock:
                            self.detection_queue.append(detection)
                        
                        print(f"Vehicle detected: {sensor_id} - {detection.lane} "
                              f"(distance: {detection.distance:.1f}cm, "
                              f"confidence: {detection.confidence:.2f})")
                
                time.sleep(self.detection_interval)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def _communication_loop(self):
        """Communication loop for sending data to central server"""
        print("Communication loop started")
        
        while self.running:
            try:
                # Collect recent detections
                detections_to_send = []
                
                with self.detection_lock:
                    if self.detection_queue:
                        detections_to_send = self.detection_queue.copy()
                        self.detection_queue.clear()
                
                # Send detections to server
                if detections_to_send:
                    self._send_detections(detections_to_send)
                
                time.sleep(self.send_interval)
                
            except Exception as e:
                print(f"Error in communication loop: {e}")
                time.sleep(5)  # Wait longer on communication errors
    
    def _send_detections(self, detections: List[VehicleDetection]):
        """Send detection data to central server"""
        try:
            # Prepare data for transmission
            detection_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_station_id': 'intersection_001',
                'detections': []
            }
            
            for detection in detections:
                detection_data['detections'].append({
                    'sensor_id': detection.sensor_id,
                    'lane': detection.lane,
                    'distance': detection.distance,
                    'timestamp': detection.timestamp.isoformat(),
                    'confidence': detection.confidence
                })
            
            # Send via UDP socket (simple and fast)
            message = json.dumps(detection_data).encode('utf-8')
            
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(1.0)  # 1 second timeout
                sock.sendto(message, (self.server_host, self.server_port))
            
            print(f"Sent {len(detections)} detections to server")
            
        except Exception as e:
            print(f"Failed to send detections: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current system status"""
        with self.detection_lock:
            queue_size = len(self.detection_queue)
        
        return {
            'running': self.running,
            'sensors_count': len(self.sensors),
            'queued_detections': queue_size,
            'sensors': {
                sensor_id: {
                    'lane': sensor.lane,
                    'trig_pin': sensor.trig_pin,
                    'echo_pin': sensor.echo_pin
                }
                for sensor_id, sensor in self.sensors.items()
            }
        }


def setup_intersection_sensors(detection_system: VehicleDetectionSystem):
    """Setup sensors for a typical 4-way intersection"""
    
    # Sensor configuration for intersection
    # Pin numbers are for Raspberry Pi GPIO (BCM numbering)
    sensor_configs = [
        # North approach
        ('sensor_north_1', 18, 24, 'north_in'),    # GPIO 18 (trig), GPIO 24 (echo)
        ('sensor_north_2', 23, 25, 'north_in'),    # Backup sensor
        
        # South approach
        ('sensor_south_1', 12, 16, 'south_in'),    # GPIO 12 (trig), GPIO 16 (echo)
        ('sensor_south_2', 20, 21, 'south_in'),    # Backup sensor
        
        # East approach
        ('sensor_east_1', 5, 6, 'east_in'),        # GPIO 5 (trig), GPIO 6 (echo)
        ('sensor_east_2', 13, 19, 'east_in'),      # Backup sensor
        
        # West approach
        ('sensor_west_1', 22, 27, 'west_in'),      # GPIO 22 (trig), GPIO 27 (echo)
        ('sensor_west_2', 17, 4, 'west_in'),       # Backup sensor
    ]
    
    for sensor_id, trig_pin, echo_pin, lane in sensor_configs:
        detection_system.add_sensor(sensor_id, trig_pin, echo_pin, lane)


def main():
    """Main function to run vehicle detection system"""
    parser = argparse.ArgumentParser(description='Vehicle Detection System with Ultrasonic Sensors')
    parser.add_argument('--server-host', default='localhost', help='Central server hostname')
    parser.add_argument('--server-port', type=int, default=8888, help='Central server port')
    parser.add_argument('--simulation', action='store_true', help='Force simulation mode')
    parser.add_argument('--duration', type=int, default=0, help='Run duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    global SIMULATION_MODE
    if args.simulation:
        SIMULATION_MODE = True
        print("Forced simulation mode enabled")
    
    print("=" * 60)
    print("VEHICLE DETECTION SYSTEM - ULTRASONIC SENSORS")
    print("=" * 60)
    print(f"Mode: {'Simulation' if SIMULATION_MODE else 'Hardware'}")
    print(f"Server: {args.server_host}:{args.server_port}")
    
    # Create detection system
    detection_system = VehicleDetectionSystem(args.server_host, args.server_port)
    
    # Setup sensors for intersection
    setup_intersection_sensors(detection_system)
    
    # Start detection
    detection_system.start_detection()
    
    try:
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely... Press Ctrl+C to stop")
            while True:
                # Show status every 10 seconds
                time.sleep(10)
                status = detection_system.get_current_status()
                print(f"Status: {status['sensors_count']} sensors, "
                      f"{status['queued_detections']} queued detections")
    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    
    finally:
        detection_system.stop_detection()
        print("Vehicle detection system stopped")


if __name__ == "__main__":
    main()
