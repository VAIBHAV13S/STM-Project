#!/usr/bin/env python3
"""
Slot-Based Scheduling Algorithm
Smart Traffic Management System - Phase 3

This script implements a slot-based scheduling system for intersection management
using predicted vehicle turning intent to assign optimal crossing time slots.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json


class MovementType(Enum):
    """Vehicle movement types at intersection"""
    STRAIGHT = "straight"
    LEFT = "left" 
    RIGHT = "right"
    UNKNOWN = "unknown"


class ConflictType(Enum):
    """Types of conflicts between movements"""
    NO_CONFLICT = 0
    CROSSING = 1      # Paths cross each other
    MERGING = 2       # Vehicles merge into same lane
    DIVERGING = 3     # Vehicles diverge from same lane


@dataclass
class Vehicle:
    """Represents a vehicle approaching the intersection"""
    id: str
    lane: str
    position: Tuple[float, float]
    speed: float
    acceleration: float
    predicted_intent: MovementType
    intent_confidence: float
    eta_to_intersection: float
    priority: int = 0  # 0=normal, 1=emergency, 2=public transport
    assigned_slot: Optional[int] = None
    waiting_time: float = 0.0
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.eta_to_intersection < other.eta_to_intersection


@dataclass
class TimeSlot:
    """Represents a time slot for intersection crossing"""
    start_time: float
    end_time: float
    assigned_vehicles: List[str] = field(default_factory=list)
    movement_type: Optional[MovementType] = None
    reserved: bool = False
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def is_available(self) -> bool:
        return not self.reserved and len(self.assigned_vehicles) == 0
    
    def can_accommodate(self, vehicle: Vehicle, conflict_matrix: Dict) -> bool:
        """Check if vehicle can be accommodated in this slot"""
        if self.reserved:
            return False
        
        # If slot is empty, it can accommodate any vehicle
        if len(self.assigned_vehicles) == 0:
            return True
        
        # Check for conflicts with already assigned vehicles
        for assigned_vehicle_id in self.assigned_vehicles:
            # This would need access to other vehicles' intents
            # For now, assume vehicles in same slot have compatible movements
            pass
        
        return True


class ConflictMatrix:
    """Manages conflict relationships between different movements"""
    
    def __init__(self):
        # Define conflict matrix: True means movements conflict
        self.conflicts = {
            (MovementType.STRAIGHT, MovementType.STRAIGHT): False,  # Parallel straight movements
            (MovementType.STRAIGHT, MovementType.LEFT): True,       # Straight vs left turn
            (MovementType.STRAIGHT, MovementType.RIGHT): False,     # Straight vs right turn (usually okay)
            (MovementType.LEFT, MovementType.LEFT): False,          # Parallel left turns
            (MovementType.LEFT, MovementType.RIGHT): True,          # Left vs right (often conflict)
            (MovementType.RIGHT, MovementType.RIGHT): False,        # Parallel right turns
        }
        
        # Direction-specific conflicts (North-South vs East-West)
        self.direction_conflicts = {
            ('north_in', 'south_in'): False,    # Same axis, opposite directions
            ('east_in', 'west_in'): False,      # Same axis, opposite directions
            ('north_in', 'east_in'): True,      # Perpendicular
            ('north_in', 'west_in'): True,      # Perpendicular
            ('south_in', 'east_in'): True,      # Perpendicular
            ('south_in', 'west_in'): True,      # Perpendicular
        }
    
    def has_conflict(self, vehicle1: Vehicle, vehicle2: Vehicle) -> bool:
        """Check if two vehicles have conflicting movements"""
        # Extract direction from lane name
        dir1 = vehicle1.lane.split('_')[0] if '_' in vehicle1.lane else vehicle1.lane
        dir2 = vehicle2.lane.split('_')[0] if '_' in vehicle2.lane else vehicle2.lane
        
        # Check direction conflicts first
        direction_pair = tuple(sorted([f"{dir1}_in", f"{dir2}_in"]))
        if direction_pair in self.direction_conflicts:
            direction_conflict = self.direction_conflicts[direction_pair]
        else:
            direction_conflict = True  # Default to conflict if unknown
        
        # If directions don't conflict, vehicles can proceed simultaneously
        if not direction_conflict:
            return False
        
        # Check movement type conflicts
        intent1_str = vehicle1.predicted_intent.value if hasattr(vehicle1.predicted_intent, 'value') else str(vehicle1.predicted_intent)
        intent2_str = vehicle2.predicted_intent.value if hasattr(vehicle2.predicted_intent, 'value') else str(vehicle2.predicted_intent)
        movement_pair = tuple(sorted([intent1_str, intent2_str]))
        return self.conflicts.get(movement_pair, True)  # Default to conflict if unknown


class SlotScheduler:
    """Main slot-based scheduling system"""
    
    def __init__(self, slot_duration: float = 3.0, lookahead_time: float = 60.0):
        self.slot_duration = slot_duration  # Duration of each time slot in seconds
        self.lookahead_time = lookahead_time  # How far ahead to schedule
        self.current_time = 0.0
        
        # Data structures
        self.time_slots: Dict[int, TimeSlot] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.conflict_matrix = ConflictMatrix()
        
        # Statistics
        self.stats = {
            'total_vehicles_processed': 0,
            'average_waiting_time': 0.0,
            'slot_utilization': 0.0,
            'conflicts_avoided': 0,
            'emergency_vehicles_processed': 0
        }
        
        # Initialize time slots
        self._initialize_slots()
    
    def _initialize_slots(self):
        """Initialize time slots for scheduling"""
        num_slots = int(self.lookahead_time / self.slot_duration)
        
        for i in range(num_slots):
            start_time = self.current_time + (i * self.slot_duration)
            end_time = start_time + self.slot_duration
            
            self.time_slots[i] = TimeSlot(
                start_time=start_time,
                end_time=end_time
            )
    
    def update_time(self, new_time: float):
        """Update current time and shift time slots"""
        time_advance = new_time - self.current_time
        self.current_time = new_time
        
        # Shift all slots by the time advance
        slots_to_remove = []
        for slot_id, slot in self.time_slots.items():
            slot.start_time += time_advance
            slot.end_time += time_advance
            
            # Remove slots that are in the past
            if slot.end_time < self.current_time:
                slots_to_remove.append(slot_id)
        
        # Remove old slots
        for slot_id in slots_to_remove:
            del self.time_slots[slot_id]
        
        # Add new slots at the end
        max_slot_id = max(self.time_slots.keys()) if self.time_slots else -1
        while len(self.time_slots) < int(self.lookahead_time / self.slot_duration):
            max_slot_id += 1
            start_time = self.current_time + (len(self.time_slots) * self.slot_duration)
            end_time = start_time + self.slot_duration
            
            self.time_slots[max_slot_id] = TimeSlot(
                start_time=start_time,
                end_time=end_time
            )
    
    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to the scheduling system"""
        self.vehicles[vehicle.id] = vehicle
        assigned_slot = self._assign_optimal_slot(vehicle)
        
        if assigned_slot is not None:
            vehicle.assigned_slot = assigned_slot
            self.time_slots[assigned_slot].assigned_vehicles.append(vehicle.id)
            
            # Set movement type for the slot if not set
            if self.time_slots[assigned_slot].movement_type is None:
                self.time_slots[assigned_slot].movement_type = vehicle.predicted_intent
            
            print(f"Vehicle {vehicle.id} assigned to slot {assigned_slot} "
                  f"({self.time_slots[assigned_slot].start_time:.1f}s - "
                  f"{self.time_slots[assigned_slot].end_time:.1f}s)")
        else:
            print(f"Warning: Could not assign slot to vehicle {vehicle.id}")
        
        self.stats['total_vehicles_processed'] += 1
        if vehicle.priority == 1:  # Emergency vehicle
            self.stats['emergency_vehicles_processed'] += 1
    
    def _assign_optimal_slot(self, vehicle: Vehicle) -> Optional[int]:
        """Find the optimal time slot for a vehicle"""
        # Calculate earliest possible slot based on ETA
        earliest_slot_time = self.current_time + vehicle.eta_to_intersection
        earliest_slot_id = max(0, int((earliest_slot_time - self.current_time) / self.slot_duration))
        
        # For emergency vehicles, try to find immediate slot or reserve one
        if vehicle.priority == 1:
            return self._assign_emergency_slot(vehicle, earliest_slot_id)
        
        # For regular vehicles, find best available slot
        best_slot = None
        min_wait_time = float('inf')
        
        # Search through available slots starting from earliest possible
        for slot_id in sorted(self.time_slots.keys()):
            if slot_id < earliest_slot_id:
                continue
            
            slot = self.time_slots[slot_id]
            
            if self._can_assign_to_slot(vehicle, slot_id):
                wait_time = max(0, slot.start_time - earliest_slot_time)
                
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    best_slot = slot_id
                
                # If we found a slot with no waiting, take it
                if wait_time == 0:
                    break
        
        return best_slot
    
    def _assign_emergency_slot(self, vehicle: Vehicle, earliest_slot_id: int) -> Optional[int]:
        """Assign slot for emergency vehicle with priority"""
        # Try to find an immediate available slot
        for slot_id in range(earliest_slot_id, min(earliest_slot_id + 3, len(self.time_slots))):
            if slot_id in self.time_slots and self.time_slots[slot_id].is_available:
                self.time_slots[slot_id].reserved = True
                return slot_id
        
        # If no immediate slot available, preempt a regular vehicle
        for slot_id in range(earliest_slot_id, min(earliest_slot_id + 5, len(self.time_slots))):
            if slot_id in self.time_slots:
                slot = self.time_slots[slot_id]
                if len(slot.assigned_vehicles) > 0:
                    # Move regular vehicles to later slots
                    vehicles_to_move = slot.assigned_vehicles.copy()
                    slot.assigned_vehicles.clear()
                    slot.reserved = True
                    
                    for veh_id in vehicles_to_move:
                        if veh_id in self.vehicles and self.vehicles[veh_id].priority == 0:
                            # Reassign regular vehicle to later slot
                            self._reassign_vehicle(veh_id)
                    
                    return slot_id
        
        return None
    
    def _can_assign_to_slot(self, vehicle: Vehicle, slot_id: int) -> bool:
        """Check if vehicle can be assigned to a specific slot"""
        if slot_id not in self.time_slots:
            return False
        
        slot = self.time_slots[slot_id]
        
        # Check if slot is reserved or full
        if slot.reserved:
            return False
        
        # Check capacity (max vehicles per slot)
        max_vehicles_per_slot = 4  # Configurable
        if len(slot.assigned_vehicles) >= max_vehicles_per_slot:
            return False
        
        # Check for conflicts with existing vehicles in slot
        for existing_veh_id in slot.assigned_vehicles:
            if existing_veh_id in self.vehicles:
                existing_vehicle = self.vehicles[existing_veh_id]
                if self.conflict_matrix.has_conflict(vehicle, existing_vehicle):
                    return False
        
        return True
    
    def _reassign_vehicle(self, vehicle_id: str):
        """Reassign a vehicle to a new slot (used for preemption)"""
        if vehicle_id not in self.vehicles:
            return
        
        vehicle = self.vehicles[vehicle_id]
        vehicle.assigned_slot = None
        
        # Find new slot
        new_slot = self._assign_optimal_slot(vehicle)
        if new_slot is not None:
            vehicle.assigned_slot = new_slot
            self.time_slots[new_slot].assigned_vehicles.append(vehicle_id)
    
    def remove_vehicle(self, vehicle_id: str):
        """Remove a vehicle from the system (when it leaves intersection)"""
        if vehicle_id not in self.vehicles:
            return
        
        vehicle = self.vehicles[vehicle_id]
        
        # Remove from assigned slot
        if vehicle.assigned_slot is not None and vehicle.assigned_slot in self.time_slots:
            slot = self.time_slots[vehicle.assigned_slot]
            if vehicle_id in slot.assigned_vehicles:
                slot.assigned_vehicles.remove(vehicle_id)
            
            # Unreserve slot if it was reserved and now empty
            if slot.reserved and len(slot.assigned_vehicles) == 0:
                slot.reserved = False
        
        # Remove from vehicles dict
        del self.vehicles[vehicle_id]
    
    def get_vehicle_speed_adjustment(self, vehicle_id: str) -> float:
        """Get recommended speed adjustment for a vehicle"""
        if vehicle_id not in self.vehicles:
            return 1.0  # No adjustment
        
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle.assigned_slot is None:
            return 0.5  # Slow down if no slot assigned
        
        slot = self.time_slots[vehicle.assigned_slot]
        time_to_slot = slot.start_time - self.current_time
        
        # Calculate required speed to arrive at optimal time
        if vehicle.eta_to_intersection > 0:
            current_speed = vehicle.speed
            required_time = time_to_slot + (self.slot_duration / 2)  # Aim for middle of slot
            
            if required_time > 0:
                distance_to_intersection = current_speed * vehicle.eta_to_intersection
                required_speed = distance_to_intersection / required_time
                speed_adjustment = required_speed / max(current_speed, 0.1)
                
                # Limit speed adjustment to reasonable range
                speed_adjustment = max(0.3, min(1.5, speed_adjustment))
                return speed_adjustment
        
        return 1.0  # No adjustment needed
    
    def get_schedule_status(self) -> Dict:
        """Get current schedule status and statistics"""
        active_slots = 0
        total_slots = len(self.time_slots)
        
        for slot in self.time_slots.values():
            if len(slot.assigned_vehicles) > 0:
                active_slots += 1
        
        slot_utilization = active_slots / total_slots if total_slots > 0 else 0
        
        # Calculate average waiting time
        total_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
        avg_waiting_time = total_waiting_time / len(self.vehicles) if self.vehicles else 0
        
        self.stats.update({
            'slot_utilization': slot_utilization,
            'average_waiting_time': avg_waiting_time,
            'active_vehicles': len(self.vehicles),
            'active_slots': active_slots,
            'total_slots': total_slots
        })
        
        return {
            'current_time': self.current_time,
            'statistics': self.stats.copy(),
            'active_vehicles': len(self.vehicles),
            'scheduled_slots': active_slots
        }
    
    def export_schedule(self, output_file: str):
        """Export current schedule to JSON file"""
        schedule_data = {
            'timestamp': datetime.now().isoformat(),
            'current_time': self.current_time,
            'statistics': self.stats,
            'time_slots': {},
            'vehicles': {}
        }
        
        # Export time slots
        for slot_id, slot in self.time_slots.items():
            schedule_data['time_slots'][str(slot_id)] = {
                'start_time': slot.start_time,
                'end_time': slot.end_time,
                'assigned_vehicles': slot.assigned_vehicles,
                'movement_type': slot.movement_type.value if slot.movement_type else None,
                'reserved': slot.reserved
            }
        
        # Export vehicles
        for veh_id, vehicle in self.vehicles.items():
            schedule_data['vehicles'][veh_id] = {
                'lane': vehicle.lane,
                'position': vehicle.position,
                'speed': vehicle.speed,
                'predicted_intent': vehicle.predicted_intent.value,
                'intent_confidence': vehicle.intent_confidence,
                'eta_to_intersection': vehicle.eta_to_intersection,
                'assigned_slot': vehicle.assigned_slot,
                'priority': vehicle.priority,
                'waiting_time': vehicle.waiting_time
            }
        
        with open(output_file, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        print(f"Schedule exported to: {output_file}")


def demo_scheduler():
    """Demonstrate the slot scheduler with sample vehicles"""
    print("=" * 60)
    print("SLOT-BASED SCHEDULER DEMONSTRATION")
    print("=" * 60)
    
    scheduler = SlotScheduler(slot_duration=3.0, lookahead_time=30.0)
    
    # Create sample vehicles
    vehicles = [
        Vehicle("car_001", "north_in", (100, 150), 15.0, 0.0, MovementType.STRAIGHT, 0.95, 5.0),
        Vehicle("car_002", "south_in", (100, 50), 12.0, 0.0, MovementType.LEFT, 0.88, 6.0),
        Vehicle("truck_003", "east_in", (150, 100), 10.0, 0.0, MovementType.RIGHT, 0.92, 8.0),
        Vehicle("car_004", "west_in", (50, 100), 14.0, 0.0, MovementType.STRAIGHT, 0.90, 4.0),
        Vehicle("emergency_005", "north_in", (100, 200), 20.0, 0.0, MovementType.STRAIGHT, 1.0, 3.0, priority=1),
    ]
    
    # Add vehicles to scheduler
    for vehicle in vehicles:
        scheduler.add_vehicle(vehicle)
    
    # Show initial schedule
    print("\nInitial Schedule:")
    status = scheduler.get_schedule_status()
    print(f"Active vehicles: {status['active_vehicles']}")
    print(f"Active slots: {status['scheduled_slots']}")
    
    # Simulate time progression
    for t in range(0, 20, 2):
        scheduler.update_time(float(t))
        
        print(f"\nTime: {t}s")
        for veh_id, vehicle in scheduler.vehicles.items():
            speed_adj = scheduler.get_vehicle_speed_adjustment(veh_id)
            print(f"  {veh_id}: Slot {vehicle.assigned_slot}, Speed adjustment: {speed_adj:.2f}x")
        
        # Remove vehicles that have passed through (simulation)
        if t >= 8:
            scheduler.remove_vehicle("car_004")
        if t >= 10:
            scheduler.remove_vehicle("emergency_005")
    
    # Export final schedule
    scheduler.export_schedule("schedule_demo.json")
    
    final_status = scheduler.get_schedule_status()
    print(f"\nFinal Statistics:")
    for key, value in final_status['statistics'].items():
        print(f"  {key}: {value}")


def process_sumo_data(data_file):
    """Process real SUMO data through the slot scheduler"""
    import pandas as pd
    
    print(f">> Loading SUMO data from: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"   Loaded {len(df)} records from {df['vehicle_id'].nunique()} unique vehicles")
        
        # Initialize scheduler
        scheduler = SlotScheduler(slot_duration=10.0, lookahead_time=120.0)
        
        # Group by vehicle and get first record for each (entry data)
        vehicle_entries = df.groupby('vehicle_id').first().reset_index()
        
        # Process first 10 vehicles for demo
        processed_count = 0
        for _, row in vehicle_entries.iterrows():
            if processed_count >= 10:  # Limit for demo
                break
                
            vehicle = Vehicle(
                id=row['vehicle_id'],
                lane=row['lane_id'],
                speed=row['speed'],
                position=row['lane_position'],
                acceleration=row.get('acceleration', 0.0),
                predicted_intent=MovementType(row['intent']) if row['intent'] in ['straight', 'left', 'right'] else MovementType.STRAIGHT,
                intent_confidence=0.8,  # Default confidence
                eta_to_intersection=row.get('distance_to_intersection', 50.0) / max(row['speed'], 1.0)  # ETA calculation
            )
            
            success = scheduler.add_vehicle(vehicle)
            if success:
                processed_count += 1
                print(f">> Added {vehicle.id}: {vehicle.predicted_intent.value} from {vehicle.lane}")
            else:
                print(f">> Failed to add {vehicle.id}")
        
        # Show final schedule
        print(f"\n>> Final Schedule Summary:")
        status = scheduler.get_schedule_status()
        for key, value in status['statistics'].items():
            print(f"   {key}: {value}")
        
        # Export schedule
        scheduler.export_schedule("schedule_sumo_data.json")
        print(f"   Schedule exported to: schedule_sumo_data.json")
        
        return True
        
    except Exception as e:
        print(f">> Error processing SUMO data: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Slot-based traffic scheduler')
    parser.add_argument('--data', help='SUMO data CSV file to process')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample vehicles')
    
    args = parser.parse_args()
    
    if args.data:
        process_sumo_data(args.data)
    elif args.demo:
        demo_scheduler()
    else:
        # Default to demo
        demo_scheduler()
