#!/usr/bin/env python3
"""
Traffic Generator for SUMO Routes
Smart Traffic Management System - Phase 1.2

This script generates various traffic patterns and route files for SUMO simulation.
It can create time-varying traffic patterns, special scenarios, and custom flows.
"""

import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import argparse


class TrafficGenerator:
    """Generates traffic patterns and SUMO route files"""
    
    def __init__(self):
        self.vehicle_types = {
            'car': {
                'accel': '2.6',
                'decel': '4.5', 
                'sigma': '0.5',
                'length': '5.0',
                'minGap': '2.5',
                'maxSpeed': '50.0',
                'color': 'yellow'
            },
            'truck': {
                'accel': '1.5',
                'decel': '3.5',
                'sigma': '0.3', 
                'length': '12.0',
                'minGap': '3.0',
                'maxSpeed': '40.0',
                'color': 'blue'
            },
            'bus': {
                'accel': '1.8',
                'decel': '4.0',
                'sigma': '0.2',
                'length': '14.0',
                'minGap': '3.5',
                'maxSpeed': '45.0',
                'color': 'red'
            },
            'emergency': {
                'accel': '3.0',
                'decel': '5.0',
                'sigma': '0.1',
                'length': '6.0',
                'minGap': '2.0',
                'maxSpeed': '60.0',
                'color': 'white'
            }
        }
        
        self.routes = {
            'north_south_straight': 'north_in south_out',
            'north_east_right': 'north_in east_out', 
            'north_west_left': 'north_in west_out',
            'south_north_straight': 'south_in north_out',
            'south_east_left': 'south_in east_out',
            'south_west_right': 'south_in west_out',
            'east_west_straight': 'east_in west_out',
            'east_north_left': 'east_in north_out',
            'east_south_right': 'east_in south_out',
            'west_east_straight': 'west_in east_out',
            'west_north_right': 'west_in north_out',
            'west_south_left': 'west_in south_out'
        }
    
    def create_route_xml(self, flows, output_file, duration=3600):
        """Create a SUMO route XML file with specified flows"""
        
        # Create root element
        root = ET.Element('routes')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        
        # Add vehicle types
        for vtype_id, attrs in self.vehicle_types.items():
            vtype = ET.SubElement(root, 'vType')
            vtype.set('id', vtype_id)
            for attr, value in attrs.items():
                vtype.set(attr, value)
        
        # Add routes
        for route_id, edges in self.routes.items():
            route = ET.SubElement(root, 'route')
            route.set('id', route_id)
            route.set('edges', edges)
        
        # Add flows
        for flow_data in flows:
            flow = ET.SubElement(root, 'flow')
            for attr, value in flow_data.items():
                flow.set(attr, str(value))
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Generated route file: {output_file}")
    
    def generate_time_varying_pattern(self, output_file):
        """Generate time-varying traffic pattern (e.g., morning rush, midday, evening rush)"""
        
        flows = []
        
        # Morning rush hour (0-1800 seconds = 0-30 minutes)
        morning_flows = [
            {'id': 'morning_north_south', 'route': 'north_south_straight', 'begin': 0, 'end': 1800, 'vehsPerHour': 1500, 'type': 'car'},
            {'id': 'morning_south_north', 'route': 'south_north_straight', 'begin': 0, 'end': 1800, 'vehsPerHour': 800, 'type': 'car'},
            {'id': 'morning_east_west', 'route': 'east_west_straight', 'begin': 0, 'end': 1800, 'vehsPerHour': 600, 'type': 'car'},
            {'id': 'morning_west_east', 'route': 'west_east_straight', 'begin': 0, 'end': 1800, 'vehsPerHour': 400, 'type': 'car'},
            {'id': 'morning_trucks', 'route': 'north_south_straight', 'begin': 0, 'end': 1800, 'vehsPerHour': 120, 'type': 'truck'},
        ]
        
        # Midday period (1800-3600 seconds = 30-60 minutes)
        midday_flows = [
            {'id': 'midday_north_south', 'route': 'north_south_straight', 'begin': 1800, 'end': 3600, 'vehsPerHour': 400, 'type': 'car'},
            {'id': 'midday_south_north', 'route': 'south_north_straight', 'begin': 1800, 'end': 3600, 'vehsPerHour': 400, 'type': 'car'},
            {'id': 'midday_east_west', 'route': 'east_west_straight', 'begin': 1800, 'end': 3600, 'vehsPerHour': 350, 'type': 'car'},
            {'id': 'midday_west_east', 'route': 'west_east_straight', 'begin': 1800, 'end': 3600, 'vehsPerHour': 350, 'type': 'car'},
            {'id': 'midday_buses', 'route': 'east_west_straight', 'begin': 1800, 'end': 3600, 'vehsPerHour': 60, 'type': 'bus'},
        ]
        
        # Evening rush hour (3600-5400 seconds = 60-90 minutes)
        evening_flows = [
            {'id': 'evening_north_south', 'route': 'north_south_straight', 'begin': 3600, 'end': 5400, 'vehsPerHour': 900, 'type': 'car'},
            {'id': 'evening_south_north', 'route': 'south_north_straight', 'begin': 3600, 'end': 5400, 'vehsPerHour': 1400, 'type': 'car'},
            {'id': 'evening_east_west', 'route': 'east_west_straight', 'begin': 3600, 'end': 5400, 'vehsPerHour': 700, 'type': 'car'},
            {'id': 'evening_west_east', 'route': 'west_east_straight', 'begin': 3600, 'end': 5400, 'vehsPerHour': 500, 'type': 'car'},
        ]
        
        flows.extend(morning_flows)
        flows.extend(midday_flows)
        flows.extend(evening_flows)
        
        self.create_route_xml(flows, output_file, duration=5400)
    
    def generate_incident_scenario(self, output_file):
        """Generate traffic pattern with incident/emergency scenario"""
        
        flows = []
        
        # Normal traffic before incident (0-1200 seconds)
        normal_flows = [
            {'id': 'normal_north_south', 'route': 'north_south_straight', 'begin': 0, 'end': 1200, 'vehsPerHour': 800, 'type': 'car'},
            {'id': 'normal_south_north', 'route': 'south_north_straight', 'begin': 0, 'end': 1200, 'vehsPerHour': 800, 'type': 'car'},
            {'id': 'normal_east_west', 'route': 'east_west_straight', 'begin': 0, 'end': 1200, 'vehsPerHour': 600, 'type': 'car'},
            {'id': 'normal_west_east', 'route': 'west_east_straight', 'begin': 0, 'end': 1200, 'vehsPerHour': 600, 'type': 'car'},
        ]
        
        # Emergency vehicles during incident
        emergency_flows = [
            {'id': 'emergency_1', 'route': 'north_south_straight', 'begin': 1200, 'end': 1300, 'vehsPerHour': 360, 'type': 'emergency'},
            {'id': 'emergency_2', 'route': 'west_east_straight', 'begin': 1250, 'end': 1350, 'vehsPerHour': 360, 'type': 'emergency'},
        ]
        
        # Reduced traffic during incident (1200-2400 seconds)
        incident_flows = [
            {'id': 'incident_north_south', 'route': 'north_south_straight', 'begin': 1200, 'end': 2400, 'vehsPerHour': 200, 'type': 'car'},
            {'id': 'incident_detour_1', 'route': 'north_east_right', 'begin': 1200, 'end': 2400, 'vehsPerHour': 400, 'type': 'car'},
            {'id': 'incident_detour_2', 'route': 'north_west_left', 'begin': 1200, 'end': 2400, 'vehsPerHour': 400, 'type': 'car'},
        ]
        
        # Recovery period (2400-3600 seconds)
        recovery_flows = [
            {'id': 'recovery_north_south', 'route': 'north_south_straight', 'begin': 2400, 'end': 3600, 'vehsPerHour': 1200, 'type': 'car'},
            {'id': 'recovery_south_north', 'route': 'south_north_straight', 'begin': 2400, 'end': 3600, 'vehsPerHour': 800, 'type': 'car'},
            {'id': 'recovery_east_west', 'route': 'east_west_straight', 'begin': 2400, 'end': 3600, 'vehsPerHour': 600, 'type': 'car'},
        ]
        
        flows.extend(normal_flows)
        flows.extend(emergency_flows)
        flows.extend(incident_flows)
        flows.extend(recovery_flows)
        
        self.create_route_xml(flows, output_file, duration=3600)
    
    def generate_asymmetric_pattern(self, output_file):
        """Generate asymmetric traffic pattern (heavy in one direction)"""
        
        flows = []
        
        # Heavy northbound traffic (commuters going to city center)
        heavy_flows = [
            {'id': 'heavy_south_north', 'route': 'south_north_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 1800, 'type': 'car'},
            {'id': 'heavy_south_east', 'route': 'south_east_left', 'begin': 0, 'end': 3600, 'vehsPerHour': 400, 'type': 'car'},
            {'id': 'heavy_south_west', 'route': 'south_west_right', 'begin': 0, 'end': 3600, 'vehsPerHour': 300, 'type': 'car'},
        ]
        
        # Light southbound traffic
        light_flows = [
            {'id': 'light_north_south', 'route': 'north_south_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 300, 'type': 'car'},
            {'id': 'light_north_east', 'route': 'north_east_right', 'begin': 0, 'end': 3600, 'vehsPerHour': 100, 'type': 'car'},
            {'id': 'light_north_west', 'route': 'north_west_left', 'begin': 0, 'end': 3600, 'vehsPerHour': 100, 'type': 'car'},
        ]
        
        # Moderate east-west traffic
        moderate_flows = [
            {'id': 'moderate_east_west', 'route': 'east_west_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 500, 'type': 'car'},
            {'id': 'moderate_west_east', 'route': 'west_east_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 600, 'type': 'car'},
            {'id': 'moderate_east_north', 'route': 'east_north_left', 'begin': 0, 'end': 3600, 'vehsPerHour': 200, 'type': 'car'},
            {'id': 'moderate_west_south', 'route': 'west_south_left', 'begin': 0, 'end': 3600, 'vehsPerHour': 250, 'type': 'car'},
        ]
        
        # Truck traffic
        truck_flows = [
            {'id': 'trucks_south_north', 'route': 'south_north_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 150, 'type': 'truck'},
            {'id': 'trucks_east_west', 'route': 'east_west_straight', 'begin': 0, 'end': 3600, 'vehsPerHour': 80, 'type': 'truck'},
        ]
        
        flows.extend(heavy_flows)
        flows.extend(light_flows)
        flows.extend(moderate_flows)
        flows.extend(truck_flows)
        
        self.create_route_xml(flows, output_file, duration=3600)
    
    def generate_random_pattern(self, output_file, min_flow=50, max_flow=1000):
        """Generate random traffic pattern for testing robustness"""
        
        flows = []
        
        for route_id in self.routes.keys():
            # Random flow rate
            flow_rate = random.randint(min_flow, max_flow)
            
            # Random vehicle type (weighted towards cars)
            vehicle_types = ['car'] * 7 + ['truck'] * 2 + ['bus'] * 1
            vehicle_type = random.choice(vehicle_types)
            
            # Random time periods
            start_time = random.randint(0, 1800)
            duration = random.randint(1200, 2400)
            end_time = min(start_time + duration, 3600)
            
            flow = {
                'id': f'random_{route_id}',
                'route': route_id,
                'begin': start_time,
                'end': end_time, 
                'vehsPerHour': flow_rate,
                'type': vehicle_type
            }
            
            flows.append(flow)
        
        # Add some additional random flows
        for i in range(random.randint(5, 15)):
            route_id = random.choice(list(self.routes.keys()))
            flow_rate = random.randint(min_flow, max_flow)
            vehicle_type = random.choice(['car', 'truck', 'bus'])
            start_time = random.randint(0, 2400)
            duration = random.randint(600, 1800)
            end_time = min(start_time + duration, 3600)
            
            flow = {
                'id': f'extra_random_{i}_{route_id}',
                'route': route_id,
                'begin': start_time,
                'end': end_time,
                'vehsPerHour': flow_rate,
                'type': vehicle_type
            }
            
            flows.append(flow)
        
        self.create_route_xml(flows, output_file, duration=3600)


def main():
    """Main function to generate various traffic patterns"""
    parser = argparse.ArgumentParser(description='Generate SUMO traffic patterns')
    parser.add_argument('--pattern', default='all', 
                       choices=['all', 'time_varying', 'incident', 'asymmetric', 'random'],
                       help='Pattern type to generate')
    parser.add_argument('--output-dir', default='routes', help='Output directory for route files')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible patterns')
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = TrafficGenerator()
    
    patterns = {
        'time_varying': generator.generate_time_varying_pattern,
        'incident': generator.generate_incident_scenario,
        'asymmetric': generator.generate_asymmetric_pattern,
        'random': generator.generate_random_pattern
    }
    
    if args.pattern == 'all':
        for pattern_name, pattern_func in patterns.items():
            output_file = os.path.join(args.output_dir, f'{pattern_name}.rou.xml')
            print(f"Generating {pattern_name} pattern...")
            pattern_func(output_file)
    else:
        if args.pattern in patterns:
            output_file = os.path.join(args.output_dir, f'{args.pattern}.rou.xml')
            print(f"Generating {args.pattern} pattern...")
            patterns[args.pattern](output_file)
        else:
            print(f"Unknown pattern: {args.pattern}")
    
    print("Traffic pattern generation completed!")


if __name__ == "__main__":
    main()
