import numpy as np
import pandas as pd
import os
from datetime import datetime


def create_sample_data():
    """
    Create sample power line fault detection dataset
    This is called by dashboard.py for cloud deployment
    """
    print("📊 Creating sample power line fault detection dataset...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate 24 hours of data (1 sample per second)
    n_samples = 24 * 60 * 60  # 24 hours * 60 minutes * 60 seconds
    time = np.arange(n_samples)

    print(f"Generating {n_samples:,} samples...")

    # Normal operation: current around 100A, voltage around 220V
    normal_current = 100 + 10 * np.sin(time / 1000) + np.random.normal(0, 3, n_samples)
    normal_voltage = 220 + 5 * np.sin(time / 2000) + np.random.normal(0, 2, n_samples)

    # Create labels: 0 = normal, 1 = fault
    labels = np.zeros(n_samples, dtype=int)

    # Add 5 random faults (for demo purposes)
    np.random.seed(42)  # For reproducible results
    n_faults = 5
    fault_durations = [30, 45, 60, 90, 120]  # Fault durations in seconds

    print("Adding simulated faults...")

    for i in range(n_faults):
        # Random fault start time (not too early, not too late)
        fault_start = np.random.randint(4 * 3600, 20 * 3600)
        fault_duration = fault_durations[i]
        fault_end = fault_start + fault_duration

        # Create fault characteristics - current spikes, voltage drops
        fault_current_spike = 300 + 100 * np.random.rand()
        fault_voltage_drop = 50 + 30 * np.random.rand()

        # Apply fault to data
        normal_current[fault_start:fault_end] = fault_current_spike + np.random.normal(0, 30, fault_duration)
        normal_voltage[fault_start:fault_end] = fault_voltage_drop + np.random.normal(0, 15, fault_duration)

        # Mark as fault
        labels[fault_start:fault_end] = 1

        # Print fault info
        fault_time = f"{fault_start // 3600}:{(fault_start % 3600) // 60:02d}"
        print(f"  Fault {i + 1}: Starts at {fault_time}, duration: {fault_duration}s")

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time,
        'current_phase_A': normal_current,
        'voltage_phase_A': normal_voltage,
        'is_fault': labels
    })

    # Add derived features
    df['power'] = df['current_phase_A'] * df['voltage_phase_A']
    df['impedance'] = df['voltage_phase_A'] / (df['current_phase_A'] + 0.001)  # Avoid division by zero

    # Save to CSV
    output_path = 'data/power_line_data.csv'
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n✅ Dataset created successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"⏱️  Time range: {df['timestamp'].min()} to {df['timestamp'].max()} seconds")
    print(f"🔴 Fault samples: {df['is_fault'].sum():,} ({df['is_fault'].sum() / len(df) * 100:.2f}%)")
    print(f"🟢 Normal samples: {(df['is_fault'] == 0).sum():,} ({(df['is_fault'] == 0).sum() / len(df) * 100:.2f}%)")

    return df


def create_minimal_data():
    """
    Create a smaller dataset for quick testing
    """
    print("Creating minimal dataset for testing...")

    n_samples = 1000
    time = np.arange(n_samples)

    # Generate data
    current = 100 + 10 * np.sin(time / 100) + np.random.normal(0, 3, n_samples)
    voltage = 220 + 5 * np.sin(time / 200) + np.random.normal(0, 2, n_samples)

    # Add a few faults
    labels = np.zeros(n_samples, dtype=int)
    fault_indices = [200, 400, 600, 800]
    for idx in fault_indices:
        current[idx:idx + 10] = 350 + np.random.normal(0, 50, 10)
        voltage[idx:idx + 10] = 80 + np.random.normal(0, 20, 10)
        labels[idx:idx + 10] = 1

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time,
        'current_phase_A': current,
        'voltage_phase_A': voltage,
        'is_fault': labels,
        'power': current * voltage,
        'impedance': voltage / (current + 0.001)
    })

    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/power_line_data.csv', index=False)

    print(f"✅ Created minimal dataset with {len(df)} samples")
    return df


def check_data_exists():
    """Check if data exists, create if missing"""
    data_path = 'data/power_line_data.csv'

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Data exists: {len(df):,} samples")
        return True
    else:
        print("❌ Data not found. Creating...")
        create_sample_data()
        return True


# For standalone testing
if __name__ == "__main__":
    print("=" * 60)
    print("POWER LINE FAULT DETECTION - DATA GENERATOR")
    print("=" * 60)

    # Ask user which dataset to create
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'minimal':
        create_minimal_data()
    else:
        create_sample_data()

    print("\n🎉 Data generation complete!")
    print("Run: streamlit run dashboard.py to start the dashboard")