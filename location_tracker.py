import numpy as np
from typing import Tuple, Dict, List
import json
from dataclasses import dataclass
from scipy import signal
import folium
from geopy.distance import geodesic
import random


@dataclass
class PowerGridNode:
    """Represents a node in the power grid"""
    node_id: str
    latitude: float
    longitude: float
    voltage_level: float  # kV
    node_type: str  # 'substation', 'transformer', 'junction', 'load'
    connected_nodes: List[str]


class FaultLocationTracker:
    """Estimate fault location in power grid"""

    def __init__(self, grid_config_file='config/grid_network.json'):
        self.grid_nodes = self.load_grid_network(grid_config_file)
        self.setup_default_network()

    def load_grid_network(self, config_file):
        """Load power grid network configuration"""
        try:
            with open(config_file, 'r') as f:
                nodes_data = json.load(f)
                return {node['node_id']: PowerGridNode(**node) for node in nodes_data}
        except FileNotFoundError:
            # Create default grid network
            return self.create_default_network()

    def create_default_network(self):
        """Create a sample power grid network"""
        nodes = {}

        # Main substations
        nodes['SUB-001'] = PowerGridNode(
            node_id='SUB-001',
            latitude=40.7128, longitude=-74.0060,
            voltage_level=230.0,
            node_type='substation',
            connected_nodes=['SUB-002', 'TFR-101', 'TFR-102']
        )

        nodes['SUB-002'] = PowerGridNode(
            node_id='SUB-002',
            latitude=40.7589, longitude=-73.9851,
            voltage_level=230.0,
            node_type='substation',
            connected_nodes=['SUB-001', 'TFR-201', 'TFR-202']
        )

        # Transformers
        nodes['TFR-101'] = PowerGridNode(
            node_id='TFR-101',
            latitude=40.7500, longitude=-73.9970,
            voltage_level=33.0,
            node_type='transformer',
            connected_nodes=['SUB-001', 'JN-101', 'JN-102']
        )

        # Add more nodes...

        # Save to config
        os.makedirs('config', exist_ok=True)
        with open('config/grid_network.json', 'w') as f:
            nodes_list = []
            for node in nodes.values():
                nodes_list.append({
                    'node_id': node.node_id,
                    'latitude': node.latitude,
                    'longitude': node.longitude,
                    'voltage_level': node.voltage_level,
                    'node_type': node.node_type,
                    'connected_nodes': node.connected_nodes
                })
            json.dump(nodes_list, f, indent=2)

        return nodes

    def impedance_based_location(self,
                                 source_node: str,
                                 measured_impedance: complex,
                                 line_impedance_per_km: complex = 0.1 + 0.05j) -> Dict:
        """
        Estimate fault location using impedance method
        Z_fault = Z_line * distance
        """

        # Calculate distance from impedance
        distance_km = abs(measured_impedance) / abs(line_impedance_per_km)

        # Get source coordinates
        source = self.grid_nodes[source_node]

        # Estimate location along the line (simplified - would need actual line paths)
        # For demo, pick a random connected node
        if source.connected_nodes:
            target_node = random.choice(source.connected_nodes)
            target = self.grid_nodes[target_node]

            # Interpolate between source and target
            ratio = min(distance_km / 10, 1.0)  # Assuming 10km max line length

            est_lat = source.latitude + (target.latitude - source.latitude) * ratio
            est_lon = source.longitude + (target.longitude - source.longitude) * ratio

            # Calculate distance from source
            source_coords = (source.latitude, source.longitude)
            est_coords = (est_lat, est_lon)
            actual_distance = geodesic(source_coords, est_coords).kilometers

            return {
                'method': 'impedance_based',
                'estimated_coordinates': (est_lat, est_lon),
                'estimated_distance_km': distance_km,
                'actual_distance_km': actual_distance,
                'source_node': source_node,
                'target_node': target_node,
                'confidence': max(0.7, 1.0 - (abs(distance_km - actual_distance) / 10))
            }

        return None

    def traveling_wave_location(self,
                                source_node: str,
                                fault_time: float,
                                wave_speed: float = 299792.458) -> Dict:
        """
        Estimate location using traveling wave method
        distance = wave_speed * time_difference / 2
        """

        # For simulation, generate random time differences
        time_diff = random.uniform(0.0001, 0.001)  # 0.1ms to 1ms

        distance_km = (wave_speed * time_diff) / 2 / 1000  # Convert to km

        source = self.grid_nodes[source_node]

        # Estimate location
        est_lat = source.latitude + random.uniform(-0.01, 0.01)
        est_lon = source.longitude + random.uniform(-0.01, 0.01)

        return {
            'method': 'traveling_wave',
            'estimated_coordinates': (est_lat, est_lon),
            'estimated_distance_km': distance_km,
            'time_difference_ms': time_diff * 1000,
            'wave_speed_km_per_s': wave_speed / 1000,
            'confidence': random.uniform(0.8, 0.95)
        }

    def multi_ended_location(self,
                             nodes_with_data: List[Tuple[str, float, complex]]) -> Dict:
        """
        Use data from multiple nodes for better accuracy
        nodes_with_data: [(node_id, fault_time, measured_impedance), ...]
        """

        if len(nodes_with_data) < 2:
            return self.impedance_based_location(nodes_with_data[0][0], nodes_with_data[0][2])

        # Simple weighted average of individual estimates
        estimates = []
        weights = []

        for node_id, fault_time, impedance in nodes_with_data:
            imp_est = self.impedance_based_location(node_id, impedance)
            wave_est = self.traveling_wave_location(node_id, fault_time)

            if imp_est:
                estimates.append(imp_est['estimated_coordinates'])
                weights.append(imp_est['confidence'])

            if wave_est:
                estimates.append(wave_est['estimated_coordinates'])
                weights.append(wave_est['confidence'])

        if not estimates:
            return None

        # Weighted average
        weights = np.array(weights) / sum(weights)
        avg_lat = sum(est[0] * w for est, w in zip(estimates, weights))
        avg_lon = sum(est[1] * w for est, w in zip(estimates, weights))

        return {
            'method': 'multi_ended',
            'estimated_coordinates': (avg_lat, avg_lon),
            'individual_estimates': estimates,
            'weights': weights.tolist(),
            'confidence': min(0.95, sum(weights) / len(weights) + 0.1)
        }

    def create_fault_location_map(self,
                                  fault_data: Dict,
                                  output_file='maps/fault_location.html') -> str:
        """Create interactive map showing fault location"""

        # Create map centered on estimated location
        est_lat, est_lon = fault_data['estimated_coordinates']

        m = folium.Map(location=[est_lat, est_lon], zoom_start=12)

        # Add grid nodes
        for node_id, node in self.grid_nodes.items():
            color = 'blue' if node.node_type == 'substation' else 'green' if node.node_type == 'transformer' else 'gray'

            folium.CircleMarker(
                location=[node.latitude, node.longitude],
                radius=8,
                popup=f"{node_id}<br>Type: {node.node_type}<br>Voltage: {node.voltage_level}kV",
                color=color,
                fill=True
            ).add_to(m)

        # Add fault location
        folium.Marker(
            location=[est_lat, est_lon],
            popup=f"⚡ FAULT LOCATION<br>Method: {fault_data['method']}<br>Confidence: {fault_data['confidence']:.2%}",
            icon=folium.Icon(color='red', icon='bolt', prefix='fa')
        ).add_to(m)

        # Add confidence circle
        folium.Circle(
            location=[est_lat, est_lon],
            radius=fault_data.get('estimated_distance_km', 1) * 1000,  # Convert km to meters
            popup=f"Estimated accuracy: ±{fault_data.get('estimated_distance_km', 1):.1f}km",
            color='red',
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

        # Save map
        os.makedirs('maps', exist_ok=True)
        m.save(output_file)

        return output_file

    def generate_location_report(self, fault_data: Dict) -> str:
        """Generate detailed location report"""

        est_lat, est_lon = fault_data['estimated_coordinates']

        report = f"""
        ⚡ POWER LINE FAULT LOCATION REPORT
        ===================================

        📍 Estimated Location:
           Latitude: {est_lat:.6f}°
           Longitude: {est_lon:.6f}°

        🔍 Location Method: {fault_data['method']}

        📊 Confidence Level: {fault_data.get('confidence', 0):.2%}

        📏 Estimated Distance from Source: {fault_data.get('estimated_distance_km', 'N/A')} km

        🗺️ Nearest Grid Nodes:
        """

        # Find nearest nodes
        fault_coords = (est_lat, est_lon)
        distances = []

        for node_id, node in self.grid_nodes.items():
            node_coords = (node.latitude, node.longitude)
            distance = geodesic(fault_coords, node_coords).kilometers
            distances.append((node_id, distance, node))

        distances.sort(key=lambda x: x[1])

        for i, (node_id, distance, node) in enumerate(distances[:5]):
            report += f"   {i + 1}. {node_id} ({node.node_type}) - {distance:.2f} km away\n"

        report += f"""

        🚨 Recommended Actions:
        1. Dispatch maintenance team to coordinates
        2. Isolate grid section: {distances[0][0]} to {distances[1][0]}
        3. Notify local authorities

        📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        return report


# Integration with dashboard
def add_location_to_dashboard():
    """Add location tracking to the dashboard"""

    location_tracker = FaultLocationTracker()

    # Add to sidebar
    st.sidebar.subheader("📍 Location Tracking")

    if st.sidebar.button("Estimate Fault Location"):
        # Simulate fault data
        fault_time = time.time()
        measured_impedance = complex(random.uniform(1, 10), random.uniform(0.5, 5))

        # Get location estimate
        location_data = location_tracker.impedance_based_location(
            source_node='SUB-001',
            measured_impedance=measured_impedance
        )

        if location_data:
            # Display on dashboard
            st.subheader("📍 Fault Location Estimation")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Latitude", f"{location_data['estimated_coordinates'][0]:.6f}°")
                st.metric("Longitude", f"{location_data['estimated_coordinates'][1]:.6f}°")
                st.metric("Confidence", f"{location_data.get('confidence', 0):.2%}")

            with col2:
                st.metric("Distance", f"{location_data.get('estimated_distance_km', 0):.2f} km")
                st.metric("Method", location_data['method'])

                # Generate map
                map_file = location_tracker.create_fault_location_map(location_data)

                # Display map
                with open(map_file, 'r', encoding='utf-8') as f:
                    map_html = f.read()

                st.components.v1.html(map_html, height=400)

            # Show report
            with st.expander("📋 View Detailed Location Report"):
                report = location_tracker.generate_location_report(location_data)
                st.text(report)