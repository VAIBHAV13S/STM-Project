#!/usr/bin/env python3
"""
Smart Traffic Management Dashboard
Smart Traffic Management System - Phase 4

This Streamlit dashboard provides real-time visualization of:
- Traffic simulation metrics
- Intent prediction accuracy  
- Slot assignment efficiency
- Vehicle flow patterns
- Real-time system monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time
import glob
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Smart Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-danger { color: #dc3545; font-weight: bold; }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: normal;
    }
</style>
""", unsafe_allow_html=True)


class TrafficDashboard:
    """Enhanced dashboard class for traffic management visualization"""
    
    def __init__(self):
        # Adjust paths for when running from visualization directory
        current_dir = Path.cwd()
        if current_dir.name == "visualization":
            self.data_dir = "../data"
            self.ml_dir = "../ml"
            self.schedule_dir = "../scheduling"
        else:
            self.data_dir = "data"
            self.ml_dir = "ml" 
            self.schedule_dir = "scheduling"
    
    def load_vehicle_data(self, file_pattern="vehicle_data_*.csv"):
        """Load vehicle data from CSV files"""
        try:
            # Handle both dashboard_data.csv and vehicle_data_*.csv patterns
            files_to_check = [
                "dashboard_data.csv",  # Copied file for dashboard
                file_pattern
            ]
            
            all_files = []
            for pattern in files_to_check:
                if pattern == "dashboard_data.csv":
                    filepath = os.path.join(".", pattern)
                    if os.path.exists(filepath):
                        all_files.append(filepath)
                else:
                    all_files.extend(glob.glob(os.path.join(self.data_dir, pattern)))
            
            if not all_files:
                return None
            
            # Load the most recent file
            latest_file = max(all_files, key=lambda x: os.path.getctime(x))
            df = pd.read_csv(latest_file)
            
            # Convert timestamp if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df, latest_file
            
        except Exception as e:
            st.error(f"Error loading vehicle data: {e}")
            return None, None
    
    def load_schedule_data(self):
        """Load schedule data from JSON files"""
        try:
            json_files = glob.glob(os.path.join(self.schedule_dir, "schedule_*.json"))
            if not json_files:
                return None, None
            
            # Load the most recent schedule file
            latest_file = max(json_files, key=lambda x: os.path.getctime(x))
            with open(latest_file, 'r') as f:
                return json.load(f), latest_file
        except Exception as e:
            st.error(f"Error loading schedule data: {e}")
            return None, None
    
    def create_real_time_metrics(self, df):
        """Create real-time performance metrics"""
        if df is None or df.empty:
            return None
        
        metrics = {
            'total_vehicles': len(df['vehicle_id'].unique()),
            'avg_speed': df['speed'].mean(),
            'max_speed': df['speed'].max(),
            'avg_waiting_time': df['waiting_time'].mean(),
            'max_waiting_time': df['waiting_time'].max(),
            'simulation_duration': df['step'].max(),
            'vehicles_per_minute': len(df['vehicle_id'].unique()) / (df['step'].max() / 60) if df['step'].max() > 0 else 0
        }
        
        return metrics
    
    def create_advanced_traffic_flow_chart(self, df):
        """Create enhanced traffic flow visualization"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Create subplots for multi-metric view
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Vehicle Count by Intent', 'Speed Distribution Over Time'),
            vertical_spacing=0.15
        )
        
        # Top plot: Vehicle flow by intent
        flow_data = df.groupby(['step', 'intent']).size().reset_index(name='count')
        
        for intent in flow_data['intent'].unique():
            intent_data = flow_data[flow_data['intent'] == intent]
            fig.add_trace(
                go.Scatter(
                    x=intent_data['step'],
                    y=intent_data['count'],
                    mode='lines+markers',
                    name=f"{intent.capitalize()}",
                    line=dict(width=3)
                ),
                row=1, col=1
            )
        
        # Bottom plot: Average speed over time
        speed_data = df.groupby('step')['speed'].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=speed_data['step'],
                y=speed_data['mean'],
                mode='lines',
                name='Average Speed',
                line=dict(color='orange', width=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add confidence bands
        fig.add_trace(
            go.Scatter(
                x=list(speed_data['step']) + list(speed_data['step'][::-1]),
                y=list(speed_data['mean'] + speed_data['std']) + list((speed_data['mean'] - speed_data['std'])[::-1]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Speed Variance',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Advanced Traffic Flow Analysis",
            height=600,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Simulation Step", row=2, col=1)
        fig.update_yaxes(title_text="Vehicle Count", row=1, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=2, col=1)
        
        return fig
    
    def create_3d_intersection_visualization(self, df):
        """Create 3D visualization of intersection traffic"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Filter to vehicles near intersection
        intersection_vehicles = df[df['distance_to_intersection'] <= 100].copy()
        
        if intersection_vehicles.empty:
            return go.Figure().add_annotation(text="No vehicles near intersection", x=0.5, y=0.5)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=intersection_vehicles['x'],
            y=intersection_vehicles['y'], 
            z=intersection_vehicles['speed'],
            mode='markers',
            marker=dict(
                size=5,
                color=intersection_vehicles['waiting_time'],
                colorscale='Viridis',
                colorbar=dict(title="Waiting Time (s)"),
                opacity=0.8
            ),
            text=intersection_vehicles['vehicle_id'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Position: (%{x:.1f}, %{y:.1f})<br>' +
                          'Speed: %{z:.1f} m/s<br>' +
                          'Waiting Time: %{marker.color:.1f} s<br>' +
                          '<extra></extra>'
        )])
        
        # Add intersection marker
        fig.add_trace(go.Scatter3d(
            x=[100], y=[100], z=[0],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            name='Intersection Center'
        ))
        
        fig.update_layout(
            title="3D Intersection Traffic Visualization",
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Speed (m/s)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_waiting_time_heatmap(self, df):
        """Create waiting time heatmap by lane and time"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Create time bins
        df_copy = df.copy()
        df_copy['time_bin'] = df_copy['step'] // 30  # 30-step bins
        
        # Calculate average waiting time by lane and time bin
        heatmap_data = df_copy.groupby(['lane_id', 'time_bin'])['waiting_time'].mean().reset_index()
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='lane_id', columns='time_bin', values='waiting_time')
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Waiting Time (s)")
        ))
        
        fig.update_layout(
            title="Waiting Time Heatmap by Lane",
            xaxis_title="Time Period (30-step bins)",
            yaxis_title="Lane ID",
            height=400
        )
        
        return fig
    
    def create_performance_gauge(self, metrics):
        """Create performance gauge charts"""
        if not metrics:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Traffic Efficiency', 'Speed Performance', 'Wait Time Score'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Traffic efficiency (vehicles per minute)
        efficiency = min(metrics['vehicles_per_minute'] * 10, 100)  # Scale to 0-100
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=efficiency,
            title={'text': "Efficiency %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}),
            row=1, col=1)
        
        # Speed performance
        speed_score = min((metrics['avg_speed'] / 15) * 100, 100)  # Normalize to 0-100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=speed_score,
            title={'text': "Speed Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"}}),
            row=1, col=2)
        
        # Wait time score (inverse)
        wait_score = max(100 - (metrics['avg_waiting_time'] * 2), 0)  # Lower is better
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=wait_score,
            title={'text': "Wait Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkorange"}}),
            row=1, col=3)
        
        fig.update_layout(height=400, title="System Performance Gauges")
        return fig
    
    def create_lane_analysis_chart(self, df):
        """Create detailed lane-by-lane analysis"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Analyze by lane
        lane_stats = df.groupby('lane_id').agg({
            'speed': ['mean', 'std'],
            'waiting_time': ['mean', 'max'],
            'vehicle_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        lane_stats.columns = ['_'.join(col).strip() for col in lane_stats.columns]
        lane_stats = lane_stats.reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Speed by Lane', 'Vehicle Count by Lane', 
                          'Average Waiting Time', 'Speed Variance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Speed by lane
        fig.add_trace(go.Bar(
            x=lane_stats['lane_id'],
            y=lane_stats['speed_mean'],
            name='Avg Speed',
            marker_color='blue'
        ), row=1, col=1)
        
        # Vehicle count by lane
        fig.add_trace(go.Bar(
            x=lane_stats['lane_id'],
            y=lane_stats['vehicle_id_nunique'],
            name='Vehicle Count',
            marker_color='green'
        ), row=1, col=2)
        
        # Waiting time by lane
        fig.add_trace(go.Bar(
            x=lane_stats['lane_id'],
            y=lane_stats['waiting_time_mean'],
            name='Avg Wait Time',
            marker_color='orange'
        ), row=2, col=1)
        
        # Speed variance by lane
        fig.add_trace(go.Bar(
            x=lane_stats['lane_id'],
            y=lane_stats['speed_std'],
            name='Speed Std Dev',
            marker_color='red'
        ), row=2, col=2)
        
        fig.update_layout(height=600, title="Lane-by-Lane Performance Analysis", showlegend=False)
        return fig
    
    def create_traffic_flow_chart(self, df):
        """Create traffic flow visualization"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Group by time step and intent
        flow_data = df.groupby(['step', 'intent']).size().reset_index(name='count')
        
        fig = px.line(
            flow_data, 
            x='step', 
            y='count', 
            color='intent',
            title="Vehicle Flow by Intent Over Time",
            labels={'step': 'Simulation Step', 'count': 'Number of Vehicles'}
        )
        
        fig.update_layout(
            xaxis_title="Simulation Step",
            yaxis_title="Number of Vehicles",
            legend_title="Intent",
            hovermode='x unified'
        )
        
        return fig
    
    def create_speed_distribution_chart(self, df):
        """Create speed distribution visualization"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        fig = px.histogram(
            df, 
            x='speed', 
            color='intent',
            nbins=30,
            title="Speed Distribution by Intent",
            labels={'speed': 'Speed (m/s)', 'count': 'Frequency'}
        )
        
        fig.update_layout(
            xaxis_title="Speed (m/s)",
            yaxis_title="Frequency",
            legend_title="Intent"
        )
        
        return fig
    
    def create_waiting_time_chart(self, df):
        """Create waiting time analysis chart"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Calculate average waiting time by step
        waiting_data = df.groupby('step')['waiting_time'].agg(['mean', 'max']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=waiting_data['step'],
            y=waiting_data['mean'],
            mode='lines',
            name='Average Waiting Time',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=waiting_data['step'],
            y=waiting_data['max'],
            mode='lines',
            name='Maximum Waiting Time',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Waiting Time Analysis",
            xaxis_title="Simulation Step",
            yaxis_title="Waiting Time (seconds)",
            legend_title="Metric",
            hovermode='x unified'
        )
        
        return fig
    
    def create_intersection_heatmap(self, df):
        """Create intersection occupancy heatmap"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Filter vehicles near intersection (within 50m)
        intersection_vehicles = df[df['distance_to_intersection'] <= 50]
        
        if intersection_vehicles.empty:
            return go.Figure().add_annotation(text="No vehicles near intersection", x=0.5, y=0.5)
        
        fig = px.density_heatmap(
            intersection_vehicles,
            x='x',
            y='y',
            nbinsx=20,
            nbinsy=20,
            title="Vehicle Density Heatmap (Near Intersection)"
        )
        
        # Add intersection marker
        fig.add_trace(go.Scatter(
            x=[100],
            y=[100],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Intersection Center'
        ))
        
        fig.update_layout(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)"
        )
        
        return fig
    
    def create_intent_accuracy_chart(self, df):
        """Create intent prediction accuracy visualization"""
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Calculate intent distribution
        intent_counts = df['intent'].value_counts()
        
        fig = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title="Intent Distribution"
        )
        
        return fig
    
    def create_slot_utilization_chart(self, schedule_data):
        """Create slot utilization visualization"""
        if schedule_data is None:
            return go.Figure().add_annotation(text="No schedule data available", x=0.5, y=0.5)
        
        # Extract slot utilization data
        slots = schedule_data.get('time_slots', {})
        
        slot_data = []
        for slot_id, slot_info in slots.items():
            slot_data.append({
                'slot_id': int(slot_id),
                'start_time': slot_info['start_time'],
                'vehicles_count': len(slot_info['assigned_vehicles']),
                'reserved': slot_info['reserved'],
                'movement_type': slot_info.get('movement_type', 'None')
            })
        
        if not slot_data:
            return go.Figure().add_annotation(text="No slot data available", x=0.5, y=0.5)
        
        slot_df = pd.DataFrame(slot_data)
        
        fig = px.bar(
            slot_df,
            x='slot_id',
            y='vehicles_count',
            color='movement_type',
            title="Slot Utilization",
            labels={'slot_id': 'Slot ID', 'vehicles_count': 'Number of Vehicles'}
        )
        
        return fig


def main():
    """Enhanced main dashboard function"""
    
    # Title with real-time status
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🚦 Smart Traffic Management Dashboard</h1>', 
                    unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**📅 Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    with col3:
        st.markdown(f"**⏰ Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize dashboard
    dashboard = TrafficDashboard()
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto-refresh with intervals
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval", 
        [10, 30, 60, 120], 
        index=1,
        format_func=lambda x: f"{x} seconds"
    )
    
    if auto_refresh:
        placeholder = st.empty()
        time.sleep(refresh_interval)
        st.rerun()
    
    # Manual controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col2:
        if st.button("📊 Export Data"):
            st.sidebar.success("Data export feature coming soon!")
    
    # View selection
    view_mode = st.sidebar.selectbox(
        "📋 Dashboard View",
        ["Overview", "Detailed Analysis", "Real-time Monitor", "Historical Trends"]
    )
    
    # Data filters
    st.sidebar.subheader("🔍 Data Filters")
    show_all_vehicles = st.sidebar.checkbox("Show All Vehicles", value=True)
    intent_filter = st.sidebar.multiselect(
        "Filter by Intent",
        ["straight", "left", "right"],
        default=["straight", "left", "right"]
    )
    
    # Load data with progress indicator
    with st.spinner("🔄 Loading traffic data..."):
        vehicle_result = dashboard.load_vehicle_data()
        schedule_result = dashboard.load_schedule_data()
        
        if vehicle_result:
            vehicle_df, data_file = vehicle_result
        else:
            vehicle_df, data_file = None, None
            
        if schedule_result:
            schedule_data, schedule_file = schedule_result
        else:
            schedule_data, schedule_file = None, None
    
    # Data status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if vehicle_df is not None:
            st.success(f"✅ Vehicle Data Loaded ({len(vehicle_df)} records)")
        else:
            st.error("❌ No Vehicle Data")
    
    with col2:
        if schedule_data is not None:
            st.success("✅ Schedule Data Loaded")
        else:
            st.warning("⚠️ No Schedule Data")
    
    with col3:
        data_age = "Unknown"
        if data_file and os.path.exists(data_file):
            mod_time = datetime.fromtimestamp(os.path.getctime(data_file))
            data_age = datetime.now() - mod_time
            st.info(f"📅 Data Age: {str(data_age).split('.')[0]}")
    
    # Main dashboard content
    if vehicle_df is not None:
        # Apply filters
        if not show_all_vehicles:
            vehicle_df = vehicle_df.head(1000)  # Limit for performance
        
        if intent_filter:
            vehicle_df = vehicle_df[vehicle_df['intent'].isin(intent_filter)]
        
        # Calculate metrics
        metrics = dashboard.create_real_time_metrics(vehicle_df)
        
        # Key Performance Indicators
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🚗 Total Vehicles", 
                metrics['total_vehicles'],
                delta=f"{metrics['vehicles_per_minute']:.1f}/min"
            )
        
        with col2:
            speed_status = "🟢" if metrics['avg_speed'] > 5 else "🟡" if metrics['avg_speed'] > 2 else "🔴"
            st.metric(
                f"{speed_status} Avg Speed", 
                f"{metrics['avg_speed']:.1f} m/s",
                delta=f"Max: {metrics['max_speed']:.1f}"
            )
        
        with col3:
            wait_status = "🟢" if metrics['avg_waiting_time'] < 10 else "🟡" if metrics['avg_waiting_time'] < 30 else "🔴"
            st.metric(
                f"{wait_status} Avg Wait Time", 
                f"{metrics['avg_waiting_time']:.1f} s",
                delta=f"Max: {metrics['max_waiting_time']:.0f}"
            )
        
        with col4:
            duration_min = metrics['simulation_duration'] / 60
            st.metric(
                "⏱️ Simulation Time", 
                f"{duration_min:.1f} min",
                delta=f"{metrics['simulation_duration']} steps"
            )
        
        with col5:
            if schedule_data:
                slot_util = schedule_data.get('statistics', {}).get('slot_utilization', 0) * 100
                util_status = "🟢" if slot_util > 70 else "🟡" if slot_util > 40 else "🔴"
                st.metric(f"{util_status} Slot Utilization", f"{slot_util:.1f}%")
            else:
                st.metric("📊 Slot Utilization", "N/A")
        
        # Performance gauges
        st.subheader("🎯 System Performance Gauges")
        gauge_fig = dashboard.create_performance_gauge(metrics)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Main content based on view mode
        if view_mode == "Overview":
            # Overview charts
            st.subheader("📈 Traffic Flow Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                flow_chart = dashboard.create_advanced_traffic_flow_chart(vehicle_df)
                st.plotly_chart(flow_chart, use_container_width=True)
            
            with col2:
                waiting_heatmap = dashboard.create_waiting_time_heatmap(vehicle_df)
                st.plotly_chart(waiting_heatmap, use_container_width=True)
            
            # Intent and spatial analysis
            st.subheader("🎯 Intent Analysis & Spatial Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                intent_data = vehicle_df['intent'].value_counts()
                intent_fig = px.pie(
                    values=intent_data.values,
                    names=intent_data.index,
                    title="Vehicle Intent Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(intent_fig, use_container_width=True)
            
            with col2:
                intersection_3d = dashboard.create_3d_intersection_visualization(vehicle_df)
                st.plotly_chart(intersection_3d, use_container_width=True)
        
        elif view_mode == "Detailed Analysis":
            # Detailed analysis view
            st.subheader("🔍 Detailed Performance Analysis")
            
            lane_analysis = dashboard.create_lane_analysis_chart(vehicle_df)
            st.plotly_chart(lane_analysis, use_container_width=True)
            
            # Statistical analysis
            st.subheader("� Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Performance Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Vehicles', 'Avg Speed', 'Max Speed', 'Avg Wait', 'Max Wait'],
                    'Value': [
                        f"{metrics['total_vehicles']:,}",
                        f"{metrics['avg_speed']:.2f} m/s",
                        f"{metrics['max_speed']:.2f} m/s",
                        f"{metrics['avg_waiting_time']:.2f} s",
                        f"{metrics['max_waiting_time']:.0f} s"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.write("**Intent Distribution**")
                intent_stats = vehicle_df['intent'].value_counts().reset_index()
                intent_stats.columns = ['Intent', 'Count']
                intent_stats['Percentage'] = (intent_stats['Count'] / intent_stats['Count'].sum() * 100).round(1)
                st.dataframe(intent_stats, hide_index=True, use_container_width=True)
        
        elif view_mode == "Real-time Monitor":
            # Real-time monitoring view
            st.subheader("� Live Traffic Monitor")
            
            # Recent vehicle activity
            recent_vehicles = vehicle_df.nlargest(50, 'step')[
                ['vehicle_id', 'step', 'speed', 'intent', 'waiting_time', 'lane_id', 'distance_to_intersection']
            ].round(2)
            
            st.write("**🚗 Recent Vehicle Activity (Last 50 vehicles)**")
            st.dataframe(recent_vehicles, use_container_width=True)
            
            # Live performance chart
            if len(vehicle_df) > 100:
                recent_data = vehicle_df.nlargest(100, 'step')
                live_flow = recent_data.groupby(['step', 'intent']).size().reset_index(name='count')
                
                live_fig = px.line(
                    live_flow, 
                    x='step', 
                    y='count', 
                    color='intent',
                    title="🔴 Live Traffic Flow (Last 100 steps)",
                    markers=True
                )
                st.plotly_chart(live_fig, use_container_width=True)
        
        elif view_mode == "Historical Trends":
            # Historical trends view
            st.subheader("📈 Historical Trends Analysis")
            
            # Time-based analysis
            if 'timestamp' in vehicle_df.columns:
                vehicle_df['hour'] = vehicle_df['timestamp'].dt.hour
                hourly_stats = vehicle_df.groupby('hour').agg({
                    'speed': 'mean',
                    'waiting_time': 'mean',
                    'vehicle_id': 'nunique'
                }).round(2)
                
                trend_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Average Speed by Hour', 'Average Wait Time by Hour', 'Vehicle Count by Hour')
                )
                
                trend_fig.add_trace(go.Bar(x=hourly_stats.index, y=hourly_stats['speed'], name='Avg Speed'), row=1, col=1)
                trend_fig.add_trace(go.Bar(x=hourly_stats.index, y=hourly_stats['waiting_time'], name='Avg Wait'), row=2, col=1)
                trend_fig.add_trace(go.Bar(x=hourly_stats.index, y=hourly_stats['vehicle_id'], name='Vehicles'), row=3, col=1)
                
                trend_fig.update_layout(height=800, title="Historical Traffic Trends")
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Timestamp data not available for historical analysis")
        
        # Schedule analysis if available
        if schedule_data:
            st.subheader("⏰ Schedule Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Active Vehicles", schedule_data.get('statistics', {}).get('active_vehicles', 0))
            
            with col2:
                st.metric("Emergency Processed", schedule_data.get('statistics', {}).get('emergency_vehicles_processed', 0))
            
            with col3:
                st.metric("Conflicts Avoided", schedule_data.get('statistics', {}).get('conflicts_avoided', 0))
            
            # Slot utilization chart
            if 'time_slots' in schedule_data:
                slots = schedule_data['time_slots']
                slot_data = []
                
                for slot_id, slot_info in slots.items():
                    slot_data.append({
                        'Slot': int(slot_id),
                        'Vehicles': len(slot_info['assigned_vehicles']),
                        'Reserved': slot_info['reserved'],
                        'Type': slot_info.get('movement_type', 'None')
                    })
                
                if slot_data:
                    slot_df = pd.DataFrame(slot_data)
                    slot_fig = px.bar(
                        slot_df, 
                        x='Slot', 
                        y='Vehicles',
                        color='Type',
                        title="Slot Utilization Analysis"
                    )
                    st.plotly_chart(slot_fig, use_container_width=True)
        
        # Export options
        st.subheader("📤 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Vehicle Data"):
                csv = vehicle_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Generate Report"):
                st.info("Report generation feature coming soon!")
        
        with col3:
            if st.button("🔄 Reset Filters"):
                st.rerun()
    
    else:
        # No data available
        st.warning("⚠️ No vehicle data available")
        
        st.markdown("""
        <div class="info-card">
        <h4>🚀 To get started:</h4>
        <ol>
        <li><strong>Run SUMO Simulation:</strong> <code>cd sumo && python runner.py --route working_rush_hour --analyze</code></li>
        <li><strong>Or run the demo:</strong> <code>python demo.py --mode demo</code></li>
        <li><strong>Then refresh this dashboard</strong></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample visualization
        st.subheader("📊 Sample Dashboard Preview")
        
        # Generate sample data for preview
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'step': range(100),
            'vehicles': np.random.poisson(5, 100),
            'avg_speed': np.random.normal(8, 2, 100),
            'waiting_time': np.random.exponential(10, 100)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_fig1 = px.line(sample_data, x='step', y='vehicles', title="Sample: Vehicle Count Over Time")
            st.plotly_chart(sample_fig1, use_container_width=True)
        
        with col2:
            sample_fig2 = px.line(sample_data, x='step', y='avg_speed', title="Sample: Average Speed Over Time")
            st.plotly_chart(sample_fig2, use_container_width=True)
    
    # Footer with system information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**📊 Dashboard:** v2.0 Enhanced")
    
    with col2:
        st.markdown(f"**🕐 Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        st.markdown(f"**📈 System Status:** {'🟢 Online' if vehicle_df is not None else '🔴 No Data'}")


if __name__ == "__main__":
    main()
