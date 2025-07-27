#!/usr/bin/env python3
"""
Advanced ETA Calculation and Path Prediction
Smart Traffic Management System - Enhanced Phase 3

This module implements sophisticated ETA calculations and path prediction
algorithms based on vehicle dynamics, traffic conditions, and machine learning.

Features:
- Dynamic ETA calculation with traffic conditions
- Path conflict prediction using graph theory
- Queue prioritization algorithms
- Real-time adjustment based on sensor data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import math
from enum import Enum


class VehicleType(Enum):
    """Vehicle type classification"""
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    EMERGENCY = "emergency"


@dataclass
class VehicleState:
    """Comprehensive vehicle state information"""
    vehicle_id: str
    position: Tuple[float, float]  # (x, y) coordinates
    speed: float  # m/s
    acceleration: float  # m/s²
    heading: float  # degrees
    vehicle_type: VehicleType
    predicted_intent: str  # straight, left, right
    intent_confidence: float
    lane: str
    distance_to_intersection: float
    timestamp: datetime
    
    # Advanced attributes
    length: float = 4.5  # Vehicle length in meters
    width: float = 2.0   # Vehicle width in meters
    max_acceleration: float = 3.0  # m/s²
    max_deceleration: float = 6.0  # m/s²
    comfort_deceleration: float = 2.5  # m/s²
    reaction_time: float = 1.5  # seconds


@dataclass
class TrafficCondition:
    """Real-time traffic condition data"""
    lane: str
    vehicle_density: float  # vehicles per meter
    average_speed: float  # m/s
    flow_rate: float  # vehicles per hour
    congestion_level: float  # 0-1 scale
    waiting_time: float  # average waiting time in seconds
    timestamp: datetime


@dataclass
class ETAResult:
    """ETA calculation result with confidence and alternatives"""
    vehicle_id: str
    estimated_arrival_time: datetime
    confidence: float
    travel_time_seconds: float
    recommended_speed: float
    alternative_etas: List[Tuple[datetime, float]]  # (time, confidence) pairs
    path_conflicts: List[str]  # Conflicting vehicle IDs
    queue_position: int
    safety_margin: float  # seconds
    calculation_method: str


class AdvancedETACalculator:
    """Advanced ETA calculation with machine learning and traffic modeling"""
    
    def __init__(self):
        # Intersection geometry (customize for specific intersection)
        self.intersection_center = (0.0, 0.0)
        self.intersection_radius = 15.0  # meters
        self.lane_width = 3.5  # meters
        
        # Traffic modeling parameters
        self.base_processing_time = 2.0  # seconds per vehicle
        self.safety_buffer = 1.0  # seconds between vehicles
        self.emergency_priority_factor = 0.5  # multiplier for emergency vehicles
        
        # Path conflict matrix (which movements conflict with each other)
        self.conflict_matrix = self._initialize_conflict_matrix()
        
        # Historical data for learning
        self.historical_eta_errors = []
        self.traffic_patterns = {}
        
        # Queue management
        self.current_queues = {
            'north': [],
            'south': [],
            'east': [],
            'west': []
        }
        
    def _initialize_conflict_matrix(self) -> Dict[Tuple[str, str], float]:
        """Initialize path conflict matrix with conflict probabilities"""
        conflicts = {}
        
        # Define movement conflicts (probability of conflict)
        movement_conflicts = [
            # Opposing straight movements (no conflict if properly timed)
            (('north', 'straight'), ('south', 'straight'), 0.0),
            (('east', 'straight'), ('west', 'straight'), 0.0),
            
            # Left turns conflicting with opposing straight
            (('north', 'left'), ('south', 'straight'), 0.9),
            (('south', 'left'), ('north', 'straight'), 0.9),
            (('east', 'left'), ('west', 'straight'), 0.9),
            (('west', 'left'), ('east', 'straight'), 0.9),
            
            # Left turns conflicting with opposing left turns
            (('north', 'left'), ('south', 'left'), 0.3),
            (('east', 'left'), ('west', 'left'), 0.3),
            
            # Cross-traffic conflicts
            (('north', 'straight'), ('east', 'straight'), 1.0),
            (('north', 'straight'), ('west', 'straight'), 1.0),
            (('south', 'straight'), ('east', 'straight'), 1.0),
            (('south', 'straight'), ('west', 'straight'), 1.0),
            
            # Right turns generally don't conflict (except with pedestrians)
            (('north', 'right'), ('east', 'straight'), 0.1),
            (('south', 'right'), ('west', 'straight'), 0.1),
            (('east', 'right'), ('south', 'straight'), 0.1),
            (('west', 'right'), ('north', 'straight'), 0.1),
        ]
        
        for movement1, movement2, conflict_prob in movement_conflicts:
            conflicts[(movement1, movement2)] = conflict_prob
            conflicts[(movement2, movement1)] = conflict_prob  # Symmetric
            
        return conflicts
    
    def calculate_base_eta(self, vehicle: VehicleState) -> float:
        """Calculate base ETA using vehicle dynamics"""
        distance = vehicle.distance_to_intersection
        current_speed = vehicle.speed
        
        if current_speed <= 0:
            # Vehicle is stopped
            return float('inf')
        
        # Basic kinematic equation with acceleration consideration
        if vehicle.acceleration != 0:
            # v² = u² + 2as, t = (v - u) / a
            # For constant acceleration to intersection
            discriminant = current_speed**2 + 2 * vehicle.acceleration * distance
            
            if discriminant < 0:
                # Vehicle is decelerating and will stop before intersection
                stop_distance = current_speed**2 / (2 * abs(vehicle.acceleration))
                if stop_distance < distance:
                    return float('inf')  # Will stop before intersection
            
            final_speed = math.sqrt(max(0, discriminant))
            travel_time = (final_speed - current_speed) / vehicle.acceleration if vehicle.acceleration != 0 else distance / current_speed
        else:
            # Constant velocity
            travel_time = distance / current_speed
        
        return max(0, travel_time)
    
    def calculate_traffic_adjusted_eta(self, vehicle: VehicleState, traffic_condition: TrafficCondition) -> float:
        """Calculate ETA adjusted for current traffic conditions"""
        base_eta = self.calculate_base_eta(vehicle)
        
        if base_eta == float('inf'):
            return base_eta
        
        # Traffic congestion adjustment
        congestion_factor = 1 + (traffic_condition.congestion_level * 2)  # Up to 3x slower
        
        # Density adjustment (more vehicles = slower movement)
        density_factor = 1 + (traffic_condition.vehicle_density * 0.1)
        
        # Speed differential adjustment
        if traffic_condition.average_speed > 0:
            speed_factor = vehicle.speed / traffic_condition.average_speed
            speed_factor = max(0.5, min(2.0, speed_factor))  # Clamp between 0.5 and 2.0
        else:
            speed_factor = 1.0
        
        adjusted_eta = base_eta * congestion_factor * density_factor / speed_factor
        
        return adjusted_eta
    
    def calculate_queue_eta(self, vehicle: VehicleState, queue_position: int) -> float:
        """Calculate ETA considering queue position and processing time"""
        # Base processing time per vehicle ahead
        queue_delay = queue_position * self.base_processing_time
        
        # Add safety buffer between vehicles
        safety_delay = queue_position * self.safety_buffer
        
        # Emergency vehicle priority
        if vehicle.vehicle_type == VehicleType.EMERGENCY:
            queue_delay *= self.emergency_priority_factor
            safety_delay *= self.emergency_priority_factor
        
        return queue_delay + safety_delay
    
    def predict_path_conflicts(self, vehicle: VehicleState, other_vehicles: List[VehicleState]) -> List[str]:
        """Predict potential path conflicts with other vehicles"""
        conflicts = []
        
        vehicle_movement = (self._get_approach_direction(vehicle.lane), vehicle.predicted_intent)
        
        for other in other_vehicles:
            if other.vehicle_id == vehicle.vehicle_id:
                continue
                
            other_movement = (self._get_approach_direction(other.lane), other.predicted_intent)
            
            # Check conflict matrix
            conflict_key = (vehicle_movement, other_movement)
            if conflict_key in self.conflict_matrix:
                conflict_prob = self.conflict_matrix[conflict_key]
                
                # Consider timing - only conflicts if vehicles arrive close in time
                vehicle_eta = self.calculate_base_eta(vehicle)
                other_eta = self.calculate_base_eta(other)
                
                time_diff = abs(vehicle_eta - other_eta)
                
                # Conflicts are likely if vehicles arrive within 5 seconds of each other
                if time_diff < 5.0 and conflict_prob > 0.5:
                    conflicts.append(other.vehicle_id)
        
        return conflicts
    
    def _get_approach_direction(self, lane: str) -> str:
        """Extract approach direction from lane name"""
        lane_lower = lane.lower()
        if 'north' in lane_lower:
            return 'north'
        elif 'south' in lane_lower:
            return 'south'
        elif 'east' in lane_lower:
            return 'east'
        elif 'west' in lane_lower:
            return 'west'
        else:
            return 'unknown'
    
    def calculate_comprehensive_eta(self, 
                                  vehicle: VehicleState, 
                                  traffic_condition: TrafficCondition,
                                  other_vehicles: List[VehicleState] = None) -> ETAResult:
        """Calculate comprehensive ETA with all factors considered"""
        
        if other_vehicles is None:
            other_vehicles = []
        
        # 1. Base ETA calculation
        base_eta = self.calculate_base_eta(vehicle)
        
        if base_eta == float('inf'):
            return ETAResult(
                vehicle_id=vehicle.vehicle_id,
                estimated_arrival_time=datetime.now() + timedelta(hours=1),  # Far future
                confidence=0.0,
                travel_time_seconds=float('inf'),
                recommended_speed=0.0,
                alternative_etas=[],
                path_conflicts=[],
                queue_position=0,
                safety_margin=0.0,
                calculation_method="stopped_vehicle"
            )
        
        # 2. Traffic-adjusted ETA
        traffic_eta = self.calculate_traffic_adjusted_eta(vehicle, traffic_condition)
        
        # 3. Queue position calculation
        approach_dir = self._get_approach_direction(vehicle.lane)
        queue_position = len(self.current_queues.get(approach_dir, []))
        queue_eta = self.calculate_queue_eta(vehicle, queue_position)
        
        # 4. Path conflict analysis
        path_conflicts = self.predict_path_conflicts(vehicle, other_vehicles)
        conflict_delay = len(path_conflicts) * 2.0  # 2 seconds per conflict
        
        # 5. Combine all factors
        total_eta = traffic_eta + queue_eta + conflict_delay
        
        # 6. Safety margin based on intent confidence
        safety_margin = (1.0 - vehicle.intent_confidence) * 3.0  # Up to 3 seconds for low confidence
        total_eta += safety_margin
        
        # 7. Calculate confidence based on various factors
        confidence = self._calculate_eta_confidence(vehicle, traffic_condition, queue_position, len(path_conflicts))
        
        # 8. Recommended speed adjustment
        recommended_speed = self._calculate_recommended_speed(vehicle, total_eta)
        
        # 9. Alternative scenarios
        alternative_etas = self._calculate_alternative_etas(vehicle, traffic_condition)
        
        # 10. Final arrival time
        arrival_time = datetime.now() + timedelta(seconds=total_eta)
        
        return ETAResult(
            vehicle_id=vehicle.vehicle_id,
            estimated_arrival_time=arrival_time,
            confidence=confidence,
            travel_time_seconds=total_eta,
            recommended_speed=recommended_speed,
            alternative_etas=alternative_etas,
            path_conflicts=path_conflicts,
            queue_position=queue_position,
            safety_margin=safety_margin,
            calculation_method="comprehensive"
        )
    
    def _calculate_eta_confidence(self, vehicle: VehicleState, traffic: TrafficCondition, 
                                queue_pos: int, conflict_count: int) -> float:
        """Calculate confidence level for ETA prediction"""
        confidence = 1.0
        
        # Reduce confidence based on various factors
        confidence *= vehicle.intent_confidence  # Intent prediction confidence
        confidence *= max(0.3, 1.0 - traffic.congestion_level * 0.5)  # Traffic congestion
        confidence *= max(0.5, 1.0 - queue_pos * 0.1)  # Queue uncertainty
        confidence *= max(0.4, 1.0 - conflict_count * 0.2)  # Path conflicts
        
        # Speed consistency factor
        if vehicle.speed > 0:
            speed_consistency = min(1.0, traffic.average_speed / vehicle.speed) if traffic.average_speed > 0 else 0.5
            confidence *= max(0.6, speed_consistency)
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_recommended_speed(self, vehicle: VehicleState, eta_seconds: float) -> float:
        """Calculate recommended speed for optimal arrival"""
        if eta_seconds <= 0 or vehicle.distance_to_intersection <= 0:
            return vehicle.speed
        
        # Ideal speed to arrive exactly at ETA
        ideal_speed = vehicle.distance_to_intersection / eta_seconds
        
        # Constrain to reasonable speed limits (0.5 to 15 m/s ~ 1.8 to 54 km/h)
        recommended_speed = max(0.5, min(15.0, ideal_speed))
        
        # Consider vehicle capabilities
        speed_change = recommended_speed - vehicle.speed
        max_achievable_change = vehicle.max_acceleration * eta_seconds
        
        if abs(speed_change) > max_achievable_change:
            if speed_change > 0:
                recommended_speed = vehicle.speed + max_achievable_change
            else:
                recommended_speed = vehicle.speed - max_achievable_change
        
        return max(0.5, recommended_speed)
    
    def _calculate_alternative_etas(self, vehicle: VehicleState, traffic: TrafficCondition) -> List[Tuple[datetime, float]]:
        """Calculate alternative ETA scenarios"""
        alternatives = []
        
        # Scenario 1: Optimistic (best case)
        optimistic_eta = self.calculate_base_eta(vehicle) * 0.8  # 20% faster
        opt_time = datetime.now() + timedelta(seconds=optimistic_eta)
        alternatives.append((opt_time, 0.2))  # Low probability
        
        # Scenario 2: Pessimistic (worst case)
        pessimistic_eta = self.calculate_base_eta(vehicle) * 2.0  # 2x slower
        pess_time = datetime.now() + timedelta(seconds=pessimistic_eta)
        alternatives.append((pess_time, 0.1))  # Very low probability
        
        # Scenario 3: No traffic delay
        no_traffic_eta = self.calculate_base_eta(vehicle)
        no_traffic_time = datetime.now() + timedelta(seconds=no_traffic_eta)
        alternatives.append((no_traffic_time, 0.3))  # Moderate probability
        
        return alternatives
    
    def update_queue_position(self, vehicle_id: str, lane: str, add: bool = True):
        """Update vehicle queue position"""
        approach_dir = self._get_approach_direction(lane)
        
        if add:
            if vehicle_id not in self.current_queues[approach_dir]:
                self.current_queues[approach_dir].append(vehicle_id)
        else:
            if vehicle_id in self.current_queues[approach_dir]:
                self.current_queues[approach_dir].remove(vehicle_id)
    
    def process_vehicle_passage(self, vehicle_id: str, lane: str, actual_time: datetime):
        """Record actual vehicle passage for learning"""
        approach_dir = self._get_approach_direction(lane)
        
        # Remove from queue
        self.update_queue_position(vehicle_id, lane, add=False)
        
        # Record for machine learning (implementation would store this data)
        passage_record = {
            'vehicle_id': vehicle_id,
            'lane': lane,
            'actual_passage_time': actual_time,
            'timestamp': datetime.now()
        }
        
        # In a full implementation, this would be stored in a database
        # for machine learning model training
    
    def get_intersection_status(self) -> Dict:
        """Get current intersection status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'queue_lengths': {direction: len(queue) for direction, queue in self.current_queues.items()},
            'total_vehicles_waiting': sum(len(queue) for queue in self.current_queues.values()),
            'conflict_matrix_size': len(self.conflict_matrix),
            'historical_records': len(self.historical_eta_errors)
        }


class ETAIntegrationService:
    """Service to integrate ETA calculations with the main traffic system"""
    
    def __init__(self):
        self.eta_calculator = AdvancedETACalculator()
        self.vehicle_states = {}
        self.traffic_conditions = {}
        
    def update_vehicle_state(self, vehicle_data: Dict):
        """Update vehicle state from sensor/CV data"""
        vehicle_state = VehicleState(
            vehicle_id=vehicle_data.get('vehicle_id', f"unknown_{datetime.now().timestamp()}"),
            position=(vehicle_data.get('x', 0), vehicle_data.get('y', 0)),
            speed=vehicle_data.get('speed', 0),
            acceleration=vehicle_data.get('acceleration', 0),
            heading=vehicle_data.get('heading', 0),
            vehicle_type=VehicleType(vehicle_data.get('vehicle_type', 'car')),
            predicted_intent=vehicle_data.get('intent', 'straight'),
            intent_confidence=vehicle_data.get('intent_confidence', 0.5),
            lane=vehicle_data.get('lane', 'unknown'),
            distance_to_intersection=vehicle_data.get('distance_to_intersection', 100),
            timestamp=datetime.now()
        )
        
        self.vehicle_states[vehicle_state.vehicle_id] = vehicle_state
    
    def update_traffic_condition(self, lane: str, condition_data: Dict):
        """Update traffic condition data"""
        condition = TrafficCondition(
            lane=lane,
            vehicle_density=condition_data.get('density', 0.1),
            average_speed=condition_data.get('avg_speed', 8.0),
            flow_rate=condition_data.get('flow_rate', 100),
            congestion_level=condition_data.get('congestion', 0.3),
            waiting_time=condition_data.get('waiting_time', 5.0),
            timestamp=datetime.now()
        )
        
        self.traffic_conditions[lane] = condition
    
    def calculate_eta_for_vehicle(self, vehicle_id: str) -> Optional[ETAResult]:
        """Calculate ETA for a specific vehicle"""
        if vehicle_id not in self.vehicle_states:
            return None
        
        vehicle = self.vehicle_states[vehicle_id]
        traffic_condition = self.traffic_conditions.get(vehicle.lane, 
            TrafficCondition(vehicle.lane, 0.1, 8.0, 100, 0.3, 5.0, datetime.now()))
        
        other_vehicles = [v for vid, v in self.vehicle_states.items() if vid != vehicle_id]
        
        return self.eta_calculator.calculate_comprehensive_eta(
            vehicle, traffic_condition, other_vehicles
        )
    
    def calculate_all_etas(self) -> Dict[str, ETAResult]:
        """Calculate ETAs for all tracked vehicles"""
        results = {}
        
        for vehicle_id in self.vehicle_states:
            eta_result = self.calculate_eta_for_vehicle(vehicle_id)
            if eta_result:
                results[vehicle_id] = eta_result
        
        return results
    
    def export_eta_data(self) -> Dict:
        """Export current ETA data for integration with other systems"""
        eta_results = self.calculate_all_etas()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'intersection_status': self.eta_calculator.get_intersection_status(),
            'vehicle_etas': {}
        }
        
        for vehicle_id, eta_result in eta_results.items():
            export_data['vehicle_etas'][vehicle_id] = {
                'estimated_arrival_time': eta_result.estimated_arrival_time.isoformat(),
                'confidence': eta_result.confidence,
                'travel_time_seconds': eta_result.travel_time_seconds,
                'recommended_speed': eta_result.recommended_speed,
                'path_conflicts': eta_result.path_conflicts,
                'queue_position': eta_result.queue_position,
                'safety_margin': eta_result.safety_margin,
                'calculation_method': eta_result.calculation_method
            }
        
        return export_data


def main():
    """Demo of advanced ETA calculation"""
    print("=" * 60)
    print("ADVANCED ETA CALCULATION DEMO")
    print("=" * 60)
    
    # Create ETA service
    eta_service = ETAIntegrationService()
    
    # Simulate vehicle data
    sample_vehicles = [
        {
            'vehicle_id': 'vehicle_001',
            'speed': 10.0,
            'acceleration': 0.0,
            'heading': 0,
            'vehicle_type': 'car',
            'intent': 'straight',
            'intent_confidence': 0.9,
            'lane': 'north_approach',
            'distance_to_intersection': 50.0
        },
        {
            'vehicle_id': 'vehicle_002',
            'speed': 8.0,
            'acceleration': -1.0,
            'heading': 180,
            'vehicle_type': 'car',
            'intent': 'left',
            'intent_confidence': 0.7,
            'lane': 'south_approach',
            'distance_to_intersection': 30.0
        }
    ]
    
    # Update vehicle states
    for vehicle_data in sample_vehicles:
        eta_service.update_vehicle_state(vehicle_data)
    
    # Update traffic conditions
    eta_service.update_traffic_condition('north_approach', {
        'density': 0.2,
        'avg_speed': 9.0,
        'congestion': 0.4
    })
    
    # Calculate ETAs
    eta_results = eta_service.calculate_all_etas()
    
    # Display results
    for vehicle_id, eta_result in eta_results.items():
        print(f"\nVehicle: {vehicle_id}")
        print(f"  ETA: {eta_result.estimated_arrival_time.strftime('%H:%M:%S')}")
        print(f"  Confidence: {eta_result.confidence:.2f}")
        print(f"  Travel Time: {eta_result.travel_time_seconds:.1f} seconds")
        print(f"  Recommended Speed: {eta_result.recommended_speed:.1f} m/s")
        print(f"  Queue Position: {eta_result.queue_position}")
        print(f"  Path Conflicts: {eta_result.path_conflicts}")
        print(f"  Safety Margin: {eta_result.safety_margin:.1f} seconds")
    
    # Export data
    export_data = eta_service.export_eta_data()
    print(f"\nExported ETA data for {len(export_data['vehicle_etas'])} vehicles")


if __name__ == "__main__":
    main()
