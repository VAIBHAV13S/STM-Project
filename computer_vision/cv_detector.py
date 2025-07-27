#!/usr/bin/env python3
"""
Computer Vision Vehicle Detection and Intent Prediction
Smart Traffic Management System - Enhanced Phase 2

This module implements computer vision-based vehicle detection using Pi Camera
and OpenCV for real-world traffic analysis and intent prediction.

Features:
- Real-time vehicle detection using OpenCV
- Intent prediction based on visual cues
- Integration with existing ML models
- Pi Camera interface for Raspberry Pi deployment
"""

# Computer vision imports with fallback handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available. Install with: pip install opencv-python")

import numpy as np
import threading
import time
import json
import socket
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import argparse
import pickle
from pathlib import Path

# Try to import Pi Camera, fallback to webcam
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Pi Camera not available, using webcam fallback")

# Machine learning imports
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available")


@dataclass
class VehicleDetectionCV:
    """Computer vision vehicle detection result"""
    vehicle_id: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    predicted_intent: str
    intent_confidence: float
    timestamp: datetime
    lane: str
    speed_estimate: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)


class ComputerVisionDetector:
    """Computer vision-based vehicle detection and intent prediction"""
    
    def __init__(self, use_pi_camera=True, resolution=(640, 480)):
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required but not installed. Install with: pip install opencv-python")
            
        self.use_pi_camera = use_pi_camera and PI_CAMERA_AVAILABLE
        self.resolution = resolution
        self.camera = None
        self.capture_thread = None
        self.running = False
        
        # Computer vision parameters
        self.vehicle_cascade = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        
        # Intent prediction model
        self.intent_model = None
        self.model_loaded = False
        
        # Vehicle tracking
        self.tracked_vehicles = {}
        self.next_vehicle_id = 1
        self.tracking_distance_threshold = 50
        
        # Data collection
        self.detection_queue = []
        self.detection_lock = threading.Lock()
        
        # Initialize camera
        self._initialize_camera()
        
        # Load ML model if available
        self._load_intent_model()
        
    def _initialize_camera(self):
        """Initialize camera (Pi Camera or webcam)"""
        try:
            if self.use_pi_camera:
                self.camera = PiCamera()
                self.camera.resolution = self.resolution
                self.camera.framerate = 30
                self.raw_capture = PiRGBArray(self.camera, size=self.resolution)
                time.sleep(2)  # Camera warm-up
                print("Pi Camera initialized")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                print("Webcam initialized")
                
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def _load_intent_model(self):
        """Load pre-trained intent prediction model"""
        try:
            model_path = Path("../ml_training/models/intent_model.pkl")
            if model_path.exists() and ML_AVAILABLE:
                self.intent_model = joblib.load(model_path)
                self.model_loaded = True
                print("Intent prediction model loaded")
            else:
                print("Intent model not found, using heuristic prediction")
        except Exception as e:
            print(f"Failed to load intent model: {e}")
    
    def _extract_vehicle_features(self, bbox, frame, motion_vector):
        """Extract features for intent prediction"""
        x, y, w, h = bbox
        
        # Basic geometric features
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Position relative to frame
        frame_h, frame_w = frame.shape[:2]
        relative_x = center_x / frame_w
        relative_y = center_y / frame_h
        
        # Motion features
        motion_magnitude = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
        motion_angle = np.arctan2(motion_vector[1], motion_vector[0])
        
        # Convert angle to degrees and normalize
        motion_angle_degrees = np.degrees(motion_angle)
        if motion_angle_degrees < 0:
            motion_angle_degrees += 360
            
        return {
            'aspect_ratio': aspect_ratio,
            'area': area,
            'relative_x': relative_x,
            'relative_y': relative_y,
            'motion_magnitude': motion_magnitude,
            'motion_angle': motion_angle_degrees,
            'bbox_width': w,
            'bbox_height': h
        }
    
    def _predict_intent_heuristic(self, features, motion_vector):
        """Heuristic-based intent prediction"""
        motion_angle = features['motion_angle']
        motion_magnitude = features['motion_magnitude']
        
        if motion_magnitude < 2:
            return "stopped", 0.5
        
        # Angle-based intent prediction
        # Assuming camera mounted at intersection looking at incoming traffic
        if 315 <= motion_angle or motion_angle <= 45:
            return "straight", 0.7
        elif 45 < motion_angle <= 135:
            return "right", 0.6
        elif 135 < motion_angle <= 225:
            return "straight", 0.7  # U-turn or backing up
        else:  # 225 < motion_angle < 315
            return "left", 0.6
    
    def _predict_intent_ml(self, features):
        """ML-based intent prediction"""
        if not self.model_loaded:
            return self._predict_intent_heuristic(features, (0, 0))
        
        try:
            # Prepare features for model
            feature_vector = np.array([[
                features['aspect_ratio'],
                features['area'],
                features['relative_x'],
                features['relative_y'],
                features['motion_magnitude'],
                features['motion_angle'],
                features['bbox_width'],
                features['bbox_height']
            ]])
            
            # Predict intent
            prediction = self.intent_model.predict(feature_vector)[0]
            probabilities = self.intent_model.predict_proba(feature_vector)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._predict_intent_heuristic(features, (0, 0))
    
    def _detect_vehicles_contours(self, frame):
        """Detect vehicles using contour analysis"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on aspect ratio (vehicles are typically wider than tall)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 4.0:
                    detections.append((x, y, w, h, area))
        
        return detections, fg_mask
    
    def _track_vehicles(self, detections, frame):
        """Track vehicles across frames and assign IDs"""
        current_vehicles = {}
        
        for detection in detections:
            x, y, w, h, area = detection
            center = (x + w//2, y + h//2)
            
            # Find closest existing vehicle
            min_distance = float('inf')
            closest_vehicle_id = None
            
            for vehicle_id, vehicle_data in self.tracked_vehicles.items():
                last_center = vehicle_data['last_center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < min_distance and distance < self.tracking_distance_threshold:
                    min_distance = distance
                    closest_vehicle_id = vehicle_id
            
            if closest_vehicle_id:
                # Update existing vehicle
                vehicle_data = self.tracked_vehicles[closest_vehicle_id]
                motion_vector = (
                    center[0] - vehicle_data['last_center'][0],
                    center[1] - vehicle_data['last_center'][1]
                )
                
                current_vehicles[closest_vehicle_id] = {
                    'bbox': (x, y, w, h),
                    'center': center,
                    'last_center': center,
                    'motion_vector': motion_vector,
                    'frame_count': vehicle_data['frame_count'] + 1,
                    'first_seen': vehicle_data['first_seen']
                }
            else:
                # New vehicle
                vehicle_id = f"cv_vehicle_{self.next_vehicle_id:04d}"
                self.next_vehicle_id += 1
                
                current_vehicles[vehicle_id] = {
                    'bbox': (x, y, w, h),
                    'center': center,
                    'last_center': center,
                    'motion_vector': (0, 0),
                    'frame_count': 1,
                    'first_seen': datetime.now()
                }
        
        self.tracked_vehicles = current_vehicles
        return current_vehicles
    
    def _determine_lane(self, center, frame_shape):
        """Determine which lane the vehicle is in based on position"""
        frame_h, frame_w = frame_shape[:2]
        x, y = center
        
        # Simple lane determination based on position
        # This should be calibrated for specific camera setup
        if x < frame_w * 0.25:
            return "left_turn_lane"
        elif x < frame_w * 0.5:
            return "straight_lane_1"
        elif x < frame_w * 0.75:
            return "straight_lane_2"
        else:
            return "right_turn_lane"
    
    def process_frame(self, frame):
        """Process a single frame for vehicle detection"""
        if frame is None:
            return [], None
        
        # Detect vehicles using contour analysis
        detections, fg_mask = self._detect_vehicles_contours(frame)
        
        # Track vehicles
        tracked_vehicles = self._track_vehicles(detections, frame)
        
        # Process each tracked vehicle
        cv_detections = []
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            if vehicle_data['frame_count'] >= 3:  # Only process vehicles seen multiple frames
                bbox = vehicle_data['bbox']
                motion_vector = vehicle_data['motion_vector']
                center = vehicle_data['center']
                
                # Extract features
                features = self._extract_vehicle_features(bbox, frame, motion_vector)
                
                # Predict intent
                intent, intent_confidence = self._predict_intent_ml(features)
                
                # Determine lane
                lane = self._determine_lane(center, frame.shape)
                
                # Estimate speed (pixels per frame, would need calibration for real speed)
                speed_estimate = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
                
                # Create detection object
                detection = VehicleDetectionCV(
                    vehicle_id=vehicle_id,
                    bbox=bbox,
                    confidence=0.8,  # Fixed confidence for contour detection
                    predicted_intent=intent,
                    intent_confidence=intent_confidence,
                    timestamp=datetime.now(),
                    lane=lane,
                    speed_estimate=speed_estimate,
                    direction_vector=motion_vector
                )
                
                cv_detections.append(detection)
        
        return cv_detections, fg_mask
    
    def start_detection(self):
        """Start the computer vision detection system"""
        if self.camera is None:
            print("Camera not available")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("Computer vision detection started")
        return True
    
    def stop_detection(self):
        """Stop the computer vision detection system"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        
        if self.camera:
            if self.use_pi_camera:
                self.camera.close()
            else:
                self.camera.release()
        
        cv2.destroyAllWindows()
        print("Computer vision detection stopped")
    
    def _capture_loop(self):
        """Main capture and processing loop"""
        print("CV capture loop started")
        
        try:
            if self.use_pi_camera:
                # Pi Camera capture loop
                for frame_data in self.camera.capture_continuous(
                    self.raw_capture, format="bgr", use_video_port=True
                ):
                    if not self.running:
                        break
                    
                    frame = frame_data.array
                    detections, fg_mask = self.process_frame(frame)
                    
                    # Store detections
                    with self.detection_lock:
                        self.detection_queue.extend(detections)
                    
                    # Display frame with detections (for debugging)
                    self._draw_detections(frame, detections)
                    
                    # Clear the stream for next frame
                    self.raw_capture.truncate(0)
                    
                    time.sleep(0.1)  # Limit frame rate
            else:
                # Webcam capture loop
                while self.running:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                    
                    detections, fg_mask = self.process_frame(frame)
                    
                    # Store detections
                    with self.detection_lock:
                        self.detection_queue.extend(detections)
                    
                    # Display frame with detections (for debugging)
                    self._draw_detections(frame, detections)
                    
                    time.sleep(0.1)  # Limit frame rate
                    
        except Exception as e:
            print(f"Error in capture loop: {e}")
    
    def _draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if detection.intent_confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw vehicle ID and intent
            label = f"{detection.vehicle_id}: {detection.predicted_intent} ({detection.intent_confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw motion vector
            center = (x + w//2, y + h//2)
            end_point = (
                center[0] + int(detection.direction_vector[0] * 3),
                center[1] + int(detection.direction_vector[1] * 3)
            )
            cv2.arrowedLine(frame, center, end_point, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Vehicle Detection', frame)
        cv2.waitKey(1)
    
    def get_detections(self):
        """Get and clear current detections"""
        with self.detection_lock:
            detections = self.detection_queue.copy()
            self.detection_queue.clear()
        return detections


class CVIntegrationServer:
    """Server to integrate CV detections with the main traffic system"""
    
    def __init__(self, port=8889):
        self.port = port
        self.cv_detector = ComputerVisionDetector()
        self.server_socket = None
        self.running = False
        
    def start_server(self):
        """Start the CV integration server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(('localhost', self.port))
        
        self.running = True
        self.cv_detector.start_detection()
        
        # Start data transmission thread
        transmission_thread = threading.Thread(target=self._transmission_loop)
        transmission_thread.daemon = True
        transmission_thread.start()
        
        print(f"CV Integration Server started on port {self.port}")
    
    def _transmission_loop(self):
        """Send CV detection data to main system"""
        while self.running:
            try:
                detections = self.cv_detector.get_detections()
                
                if detections:
                    # Prepare data for transmission
                    detection_data = {
                        'timestamp': datetime.now().isoformat(),
                        'source': 'computer_vision',
                        'detections': []
                    }
                    
                    for detection in detections:
                        detection_data['detections'].append({
                            'vehicle_id': detection.vehicle_id,
                            'bbox': detection.bbox,
                            'confidence': detection.confidence,
                            'predicted_intent': detection.predicted_intent,
                            'intent_confidence': detection.intent_confidence,
                            'lane': detection.lane,
                            'speed_estimate': detection.speed_estimate,
                            'timestamp': detection.timestamp.isoformat()
                        })
                    
                    # Send to main traffic system (port 8888)
                    message = json.dumps(detection_data).encode('utf-8')
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        sock.sendto(message, ('localhost', 8888))
                    
                    print(f"Sent {len(detections)} CV detections to main system")
                
                time.sleep(1.0)  # Send data every second
                
            except Exception as e:
                print(f"Error in transmission loop: {e}")
                time.sleep(5)
    
    def stop_server(self):
        """Stop the CV integration server"""
        self.running = False
        self.cv_detector.stop_detection()
        
        if self.server_socket:
            self.server_socket.close()
        
        print("CV Integration Server stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Computer Vision Vehicle Detection')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of Pi Camera')
    parser.add_argument('--port', type=int, default=8889, help='Server port')
    parser.add_argument('--duration', type=int, default=0, help='Run duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPUTER VISION VEHICLE DETECTION")
    print("=" * 60)
    
    # Create CV integration server
    cv_server = CVIntegrationServer(args.port)
    
    # Override camera preference if specified
    if args.webcam:
        cv_server.cv_detector.use_pi_camera = False
        cv_server.cv_detector._initialize_camera()
    
    try:
        cv_server.start_server()
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely... Press Ctrl+C to stop")
            while True:
                time.sleep(10)
                print("CV detection system running...")
    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    
    finally:
        cv_server.stop_server()


if __name__ == "__main__":
    main()
