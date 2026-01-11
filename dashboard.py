import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import winsound
import threading
import time
import os
# <-- Add if missing
# ========== CLOUD-READY FUNCTIONS ==========

def ensure_data_exists():
    """Create data folder and sample data if missing"""
    data_path = 'data/power_line_data.csv'

    # Create data folder if it doesn't exist
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(data_path):
        try:
            from create_dataset import create_sample_data
            create_sample_data()
            st.success("✅ Sample data created successfully!")
        except:
            # Create minimal data if import fails
            df = pd.DataFrame({
                'timestamp': range(1000),
                'current_phase_A': 100 + 10 * np.random.randn(1000),
                'voltage_phase_A': 220 + 5 * np.random.randn(1000),
                'is_fault': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
            })
            df.to_csv(data_path, index=False)
            st.info("📊 Created minimal sample data")


def ensure_models_exist():
    """Create models folder if missing"""
    os.makedirs('models', exist_ok=True)

    # You can add model creation logic here if needed
    # Or your existing code will handle it
    pass


def load_model_safely():
    """Load model with error handling"""
    try:
        model = joblib.load('models/fault_detection_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        st.warning("⚠️ Models not found. Running in simulation mode.")
        return None, None
# ==================== SOUND ALERT FUNCTION ====================
def play_alert_sound(fault_type="general"):
    """Play different sounds based on fault severity"""
    try:
        if fault_type == "critical":
            # High-pitched alarm for critical faults
            for _ in range(3):
                winsound.Beep(2000, 500)  # Frequency: 2000Hz, Duration: 500ms
                time.sleep(0.2)
        else:
            # Single beep for minor faults
            winsound.Beep(1000, 1000)  # Frequency: 1000Hz, Duration: 1000ms
    except:
        pass  # Silently fail if sound doesn't work (e.g., on Mac/Linux)

# ==================== END SOUND FUNCTION ====================
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Power Line Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .fault-alert {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        padding: 1rem;
        border-radius: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .normal-status {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Load trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fault_detection_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        st.error("Model files not found. Please run the ML notebook first.")
        return None, None


# Function to generate simulated real-time data
def generate_live_data(include_fault=False, fault_duration=60):
    """Generate simulated power line data"""
    n_samples = 500
    t = np.linspace(0, 50, n_samples)

    # Base signals
    current_base = 100 + 15 * np.sin(t / 2) + np.random.normal(0, 5, n_samples)
    voltage_base = 220 + 10 * np.sin(t / 3) + np.random.normal(0, 3, n_samples)

    if include_fault:
        # Create fault in the middle
        fault_start = n_samples // 3
        fault_end = fault_start + fault_duration

        # Modify signals during fault
        current_base[fault_start:fault_end] = (
                350 + 100 * np.sin(t[fault_start:fault_end] / 5) +
                np.random.normal(0, 50, fault_end - fault_start)
        )
        voltage_base[fault_start:fault_end] = (
                80 + 40 * np.sin(t[fault_start:fault_end] / 5) +
                np.random.normal(0, 25, fault_end - fault_start)
        )

        is_fault = np.zeros(n_samples)
        is_fault[fault_start:fault_end] = 1
    else:
        is_fault = np.zeros(n_samples)

    # Calculate derived metrics
    power = current_base * voltage_base
    impedance = voltage_base / (current_base + 0.001)

    return {
        'time': t,
        'current': current_base,
        'voltage': voltage_base,
        'power': power,
        'impedance': impedance,
        'is_fault': is_fault
    }


# Main dashboard function
def main():
    # Title and description
    st.markdown('<h1 class="main-header">⚡ AI-Powered Power Line Fault Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
    Real-time monitoring and intelligent fault detection using machine learning
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, scaler = load_model()

    if model is None or scaler is None:
        st.stop()

    # Sidebar for controls
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")

        # Simulation mode
        st.markdown("#### Simulation Mode")
        simulation_mode = st.selectbox(
            "Select Simulation Mode",
            ["Normal Operation", "Random Faults", "Manual Fault Control"]
        )

        # Fault probability slider
        if simulation_mode == "Random Faults":
            fault_probability = st.slider("Fault Probability (%)", 0, 100, 10)
        else:
            fault_probability = 0

        # Manual fault control
        if simulation_mode == "Manual Fault Control":
            trigger_fault = st.button("🔴 Trigger Fault Now", type="primary", use_container_width=True)
            clear_fault = st.button("🟢 Clear Fault", use_container_width=True)
        else:
            trigger_fault = False
            clear_fault = False

        # Display settings
        st.markdown("#### Display Settings")
        chart_speed = st.slider("Chart Update Speed (seconds)", 1, 10, 3)
        samples_to_display = st.slider("Samples to Display", 100, 1000, 500)

        # Model info
        st.markdown("---")
        st.markdown("#### 📊 Model Information")
        st.info(f"Model: Random Forest\nAccuracy: ~98%\nLast Updated: {datetime.now().strftime('%Y-%m-%d')}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📈 Live Power Line Monitoring</h2>', unsafe_allow_html=True)

        # Create placeholders for live updates
        chart_placeholder = st.empty()
        status_placeholder = st.empty()

        # Fault history
        st.markdown('<h2 class="sub-header">📋 Fault History Log</h2>', unsafe_allow_html=True)
        log_placeholder = st.empty()

    with col2:
        st.markdown('<h2 class="sub-header">📊 System Metrics</h2>', unsafe_allow_html=True)

        # Create metric placeholders
        metric1 = st.empty()
        metric2 = st.empty()
        metric3 = st.empty()
        metric4 = st.empty()

        # Fault statistics
        st.markdown('<h2 class="sub-header">🔍 Fault Statistics</h2>', unsafe_allow_html=True)
        stats_placeholder = st.empty()

    # Initialize session state for fault log
    if 'fault_log' not in st.session_state:
        st.session_state.fault_log = []
    if 'fault_count' not in st.session_state:
        st.session_state.fault_count = 0
    if 'normal_count' not in st.session_state:
        st.session_state.normal_count = 0

    # Main simulation loop
    simulation_active = True
    fault_triggered = False

    while simulation_active:
        # Determine if we should generate a fault
        if simulation_mode == "Manual Fault Control":
            if trigger_fault and not fault_triggered:
                fault_triggered = True
            if clear_fault:
                fault_triggered = False
            generate_fault = fault_triggered
        elif simulation_mode == "Random Faults":
            generate_fault = np.random.rand() < (fault_probability / 100)
        else:
            generate_fault = False

        # Generate data
        data = generate_live_data(include_fault=generate_fault)

        # Prepare features for prediction
        features = np.array([
            data['current'][-1],
            data['voltage'][-1],
            data['power'][-1],
            data['impedance'][-1]
        ]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Update counters
        if prediction == 1:
            st.session_state.fault_count += 1
            # Log fault if it's new
            current_time = datetime.now().strftime("%H:%M:%S")
            if len(st.session_state.fault_log) == 0 or not st.session_state.fault_log[-1].startswith("🔴"):
                log_entry = f"🔴 Fault detected at {current_time} (Confidence: {probability:.2%})"
                st.session_state.fault_log.append(log_entry)
        else:
            st.session_state.normal_count += 1

        # Update metrics
        with col2:
            with metric1.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#3B82F6;">Current</h3>
                    <p style="font-size: 1.8rem; margin:0.5rem 0; font-weight:bold;">{data['current'][-1]:.1f} A</p>
                    <p style="margin:0; color:#6B7280;">Phase A</p>
                </div>
                """, unsafe_allow_html=True)

            with metric2.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#10B981;">Voltage</h3>
                    <p style="font-size: 1.8rem; margin:0.5rem 0; font-weight:bold;">{data['voltage'][-1]:.1f} V</p>
                    <p style="margin:0; color:#6B7280;">Phase A</p>
                </div>
                """, unsafe_allow_html=True)

            with metric3.container():
                fault_color = "#EF4444" if prediction == 1 else "#10B981"
                fault_text = "FAULT DETECTED" if prediction == 1 else "SYSTEM NORMAL"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:{fault_color};">Status</h3>
                    <p style="font-size: 1.8rem; margin:0.5rem 0; font-weight:bold;">{fault_text}</p>
                    <p style="margin:0; color:#6B7280;">Confidence: {probability:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            with metric4.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#8B5CF6;">Power</h3>
                    <p style="font-size: 1.8rem; margin:0.5rem 0; font-weight:bold;">{data['power'][-1]:.1f} kW</p>
                    <p style="margin:0; color:#6B7280;">Active Power</p>
                </div>
                """, unsafe_allow_html=True)

        # Update status display
        with col1:
            if prediction == 1:
                status_placeholder.markdown(f"""
                <div class="fault-alert">
                    <h2 style="color:#DC2626; margin:0;">🚨 FAULT ALERT!</h2>
                    <p style="margin:0.5rem 0;">High probability of fault detected in power line.</p>
                    <p style="margin:0;">Recommended action: Dispatch maintenance team.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f"""
                <div class="normal-status">
                    <h2 style="color:#10B981; margin:0;">✅ SYSTEM NORMAL</h2>
                    <p style="margin:0.5rem 0;">All parameters within safe operating limits.</p>
                </div>
                """, unsafe_allow_html=True)

        # Update charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Current Phase A', 'Voltage Phase A',
                            'Active Power', 'Impedance'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Add traces
        fig.add_trace(
            go.Scatter(x=data['time'][-samples_to_display:],
                       y=data['current'][-samples_to_display:],
                       mode='lines',
                       name='Current',
                       line=dict(color='#3B82F6', width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=data['time'][-samples_to_display:],
                       y=data['voltage'][-samples_to_display:],
                       mode='lines',
                       name='Voltage',
                       line=dict(color='#10B981', width=2)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=data['time'][-samples_to_display:],
                       y=data['power'][-samples_to_display:],
                       mode='lines',
                       name='Power',
                       line=dict(color='#8B5CF6', width=2)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=data['time'][-samples_to_display:],
                       y=data['impedance'][-samples_to_display:],
                       mode='lines',
                       name='Impedance',
                       line=dict(color='#F59E0B', width=2)),
            row=2, col=2
        )

        # Highlight fault regions if present
        if any(data['is_fault'][-samples_to_display:]):
            fault_indices = np.where(data['is_fault'][-samples_to_display:] == 1)[0]
            if len(fault_indices) > 0:
                fault_times = data['time'][-samples_to_display:][fault_indices]
                for i in range(4):
                    fig.add_vrect(
                        x0=fault_times[0], x1=fault_times[-1],
                        fillcolor="red", opacity=0.2,
                        line_width=0,
                        row=(i // 2) + 1, col=(i % 2) + 1
                    )

        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            template="plotly_white",
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Current (A)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
        fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
        fig.update_yaxes(title_text="Impedance (Ω)", row=2, col=2)

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update fault log
        with col1:
            with log_placeholder.container():
                if st.session_state.fault_log:
                    st.markdown("### Recent Faults")
                    for log_entry in st.session_state.fault_log[-5:]:  # Show last 5
                        st.write(log_entry)
                else:
                    st.info("No faults detected. System operating normally.")

        # Update statistics
        with col2:
            with stats_placeholder.container():
                total_samples = st.session_state.fault_count + st.session_state.normal_count
                if total_samples > 0:
                    fault_percentage = (st.session_state.fault_count / total_samples) * 100
                    st.metric("Total Faults Detected", st.session_state.fault_count)
                    st.metric("Fault Detection Rate", f"{fault_percentage:.2f}%")
                    st.metric("System Uptime", f"{(st.session_state.normal_count / total_samples * 100):.1f}%")
                else:
                    st.info("Waiting for data...")

        # Wait before next update
        time.sleep(chart_speed)

        # Check if we should stop (for demo purposes)
        # In production, this would run continuously
        # simulation_active = False  # Uncomment to run once for testing
if __name__ == "__main__":
    # ========== CLOUD SETUP ==========
    # Create necessary folders for cloud deployment
    import os

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Check if data exists, create if missing
    if not os.path.exists('data/power_line_data.csv'):
        print("📊 Setting up for first run...")
        try:
            # Try to import your data generator
            from create_dataset import create_sample_data
            create_sample_data()
        except Exception as e:
            print(f"⚠️ Could not import create_dataset: {e}")
            # Create minimal data if import fails
            import pandas as pd
            import numpy as np

            df = pd.DataFrame({
                'timestamp': range(1000),
                'current_phase_A': 100 + 10 * np.random.randn(1000),
                'voltage_phase_A': 220 + 5 * np.random.randn(1000),
                'is_fault': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
            })
            df.to_csv('data/power_line_data.csv', index=False)
            print("✅ Created minimal dataset")

    # ========== START DASHBOARD ==========
    main()