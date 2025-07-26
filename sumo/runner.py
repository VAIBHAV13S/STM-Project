#!/usr/bin/env python3
"""
SUMO Traffic Simulation with TraCI Data Collection
Smart Traffic Management System - Phase 1

This script runs a SUMO simulation and collects vehicle data for ML training.
It tracks vehicle positions, speeds, lanes, routes, and derives turning intent.
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime
import pandas as pd

# TraCI import
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib


class TrafficDataCollector:
    """Collects and processes vehicle data from SUMO simulation"""
    
    def __init__(self, output_file="vehicle_data.csv"):
        self.output_file = output_file
        self.data_rows = []
        self.step_count = 0
        
        # Initialize CSV file with headers
        self.headers = [
            'timestamp', 'step', 'vehicle_id', 'x', 'y', 'speed', 'acceleration',
            'lane_id', 'lane_position', 'route_id', 'route_edges', 'intent',
            'distance_to_intersection', 'waiting_time', 'angle', 'vehicle_type'
        ]
        
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
    
    def extract_intent_from_route(self, route_edges):
        """Extract turning intent from route edges"""
        if len(route_edges) < 2:
            return 'unknown'
        
        start_edge = route_edges[0]
        end_edge = route_edges[-1]
        
        # Define movement patterns based on edge names
        movements = {
            ('north_in', 'south_out'): 'straight',
            ('north_in', 'east_out'): 'right',
            ('north_in', 'west_out'): 'left',
            ('south_in', 'north_out'): 'straight', 
            ('south_in', 'east_out'): 'left',
            ('south_in', 'west_out'): 'right',
            ('east_in', 'west_out'): 'straight',
            ('east_in', 'north_out'): 'left',
            ('east_in', 'south_out'): 'right',
            ('west_in', 'east_out'): 'straight',
            ('west_in', 'north_out'): 'right',
            ('west_in', 'south_out'): 'left'
        }
        
        return movements.get((start_edge, end_edge), 'unknown')
    
    def calculate_distance_to_intersection(self, x, y):
        """Calculate Euclidean distance to intersection center (100, 100)"""
        intersection_x, intersection_y = 100.0, 100.0
        return ((x - intersection_x) ** 2 + (y - intersection_y) ** 2) ** 0.5
    
    def collect_vehicle_data(self):
        """Collect data for all vehicles in current simulation step"""
        current_time = datetime.now().isoformat()
        
        for veh_id in traci.vehicle.getIDList():
            try:
                # Basic position and movement data
                position = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_position = traci.vehicle.getLanePosition(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                vehicle_type = traci.vehicle.getTypeID(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                
                # Route information
                route_id = traci.vehicle.getRouteID(veh_id)
                route_edges = traci.vehicle.getRoute(veh_id)
                
                # Derive intent from route
                intent = self.extract_intent_from_route(route_edges)
                
                # Calculate distance to intersection
                distance_to_intersection = self.calculate_distance_to_intersection(
                    position[0], position[1]
                )
                
                # Create data row
                row = [
                    current_time,
                    self.step_count,
                    veh_id,
                    position[0],
                    position[1],
                    speed,
                    acceleration,
                    lane_id,
                    lane_position,
                    route_id,
                    '|'.join(route_edges),  # Join route edges with pipe separator
                    intent,
                    distance_to_intersection,
                    waiting_time,
                    angle,
                    vehicle_type
                ]
                
                self.data_rows.append(row)
                
            except Exception as e:
                print(f"Error collecting data for vehicle {veh_id}: {e}")
    
    def save_data_batch(self, batch_size=1000):
        """Save collected data to CSV file in batches"""
        if len(self.data_rows) >= batch_size:
            with open(self.output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.data_rows)
            
            print(f"Saved batch of {len(self.data_rows)} records at step {self.step_count}")
            self.data_rows = []
    
    def save_remaining_data(self):
        """Save any remaining data at end of simulation"""
        if self.data_rows:
            with open(self.output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.data_rows)
            print(f"Saved final batch of {len(self.data_rows)} records")


def run_simulation(route_file, network_file, steps=3600, gui=False):
    """Run SUMO simulation with data collection"""
    
    # SUMO configuration
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-n", network_file,
        "-r", route_file,
        "--waiting-time-memory", "300",
        "--time-to-teleport", "300",
        "--no-step-log",
        "--no-warnings"
    ]
    
    # Output file based on route file name
    route_name = os.path.basename(route_file).replace('.rou.xml', '')
    output_file = f"../data/vehicle_data_{route_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Starting SUMO simulation...")
    print(f"Route file: {route_file}")
    print(f"Network file: {network_file}")
    print(f"Output file: {output_file}")
    print(f"Simulation steps: {steps}")
    
    # Initialize data collector
    data_collector = TrafficDataCollector(output_file)
    
    try:
        # Start TraCI
        traci.start(sumo_cmd)
        
        # Run simulation
        step = 0
        while step < steps and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect vehicle data every step
            data_collector.step_count = step
            data_collector.collect_vehicle_data()
            
            # Save data in batches to prevent memory issues
            data_collector.save_data_batch(batch_size=500)
            
            # Progress indicator
            if step % 100 == 0:
                num_vehicles = len(traci.vehicle.getIDList())
                print(f"Step {step}: {num_vehicles} vehicles active")
            
            step += 1
        
        # Save any remaining data
        data_collector.save_remaining_data()
        
        print(f"Simulation completed! Data saved to {output_file}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        traci.close()
    
    return output_file


def analyze_collected_data(data_file):
    """Perform basic analysis of collected data"""
    try:
        df = pd.read_csv(data_file)
        
        print(f"\n=== Data Analysis for {data_file} ===")
        print(f"Total records: {len(df)}")
        print(f"Unique vehicles: {df['vehicle_id'].nunique()}")
        print(f"Simulation duration: {df['step'].max()} steps")
        
        print(f"\nIntent distribution:")
        intent_counts = df['intent'].value_counts()
        for intent, count in intent_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {intent}: {count} ({percentage:.1f}%)")
        
        print(f"\nVehicle type distribution:")
        type_counts = df['vehicle_type'].value_counts()
        for vtype, count in type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {vtype}: {count} ({percentage:.1f}%)")
        
        print(f"\nSpeed statistics:")
        print(f"  Mean: {df['speed'].mean():.2f} m/s")
        print(f"  Max: {df['speed'].max():.2f} m/s")
        print(f"  Std: {df['speed'].std():.2f} m/s")
        
        print(f"\nWaiting time statistics:")
        print(f"  Mean: {df['waiting_time'].mean():.2f} s")
        print(f"  Max: {df['waiting_time'].max():.2f} s")
        
    except Exception as e:
        print(f"Error analyzing data: {e}")


def main():
    """Main function to run simulations for different traffic patterns"""
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with data collection')
    parser.add_argument('--route', default='all', help='Route file (rush_hour, balanced, sparse, or all)')
    parser.add_argument('--steps', type=int, default=3600, help='Number of simulation steps')
    parser.add_argument('--gui', action='store_true', help='Run with SUMO GUI')
    parser.add_argument('--analyze', action='store_true', help='Analyze collected data after simulation')
    
    args = parser.parse_args()
    
    # Define route files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    network_file = os.path.join(base_dir, "intersection.net.xml")
    
    route_files = {
        'rush_hour': os.path.join(base_dir, "routes", "rush_hour.rou.xml"),
        'balanced': os.path.join(base_dir, "routes", "balanced.rou.xml"),
        'sparse': os.path.join(base_dir, "routes", "sparse.rou.xml"),
        'working_rush_hour': os.path.join(base_dir, "routes", "working_rush_hour.rou.xml")
    }
    
    # Check if network file exists
    if not os.path.exists(network_file):
        print(f"Network file not found: {network_file}")
        return
    
    output_files = []
    
    if args.route == 'all':
        # Run all route scenarios
        for route_name, route_file in route_files.items():
            if os.path.exists(route_file):
                print(f"\n{'='*50}")
                print(f"Running simulation: {route_name}")
                print(f"{'='*50}")
                output_file = run_simulation(route_file, network_file, args.steps, args.gui)
                output_files.append(output_file)
            else:
                print(f"Route file not found: {route_file}")
    else:
        # Run specific route scenario
        if args.route in route_files:
            route_file = route_files[args.route]
            if os.path.exists(route_file):
                output_file = run_simulation(route_file, network_file, args.steps, args.gui)
                output_files.append(output_file)
            else:
                print(f"Route file not found: {route_file}")
        else:
            print(f"Unknown route: {args.route}")
            print(f"Available routes: {list(route_files.keys())} or 'all'")
    
    # Analyze collected data if requested
    if args.analyze:
        for output_file in output_files:
            if os.path.exists(output_file):
                analyze_collected_data(output_file)


if __name__ == "__main__":
    main()
