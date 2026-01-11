import numpy as np
import pandas as pd
import os

print("🔧 Creating power line fault dataset...")

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate 24 hours of data (1 sample per second)
print("Generating 24 hours of power data...")
n_samples = 24 * 60 * 60  # 24 hours * 60 minutes * 60 seconds
time = np.arange(n_samples)

# Normal operation: current around 100A, voltage around 220V
normal_current = 100 + 10 * np.sin(time / 1000) + np.random.normal(0, 3, n_samples)
normal_voltage = 220 + 5 * np.sin(time / 2000) + np.random.normal(0, 2, n_samples)

# Create labels: 0 = normal, 1 = fault
labels = np.zeros(n_samples, dtype=int)

# Add 5 random faults
np.random.seed(42)  # For reproducible results
n_faults = 5
fault_durations = [30, 45, 60, 90, 120]  # Fault durations in seconds

for i in range(n_faults):
    # Random fault start time (not too early, not too late)
    fault_start = np.random.randint(4 * 3600, 20 * 3600)
    fault_duration = fault_durations[i]
    fault_end = fault_start + fault_duration

    print(f"Fault {i + 1}: Starts at {fault_start // 3600}:{(fault_start % 3600) // 60:02d}, "
          f"duration: {fault_duration} seconds")

    # Create fault characteristics
    # Current spikes, voltage drops
    fault_current_spike = 300 + 100 * np.random.rand()
    fault_voltage_drop = 50 + 30 * np.random.rand()

    # Apply fault to data
    normal_current[fault_start:fault_end] = fault_current_spike + np.random.normal(0, 30, fault_duration)
    normal_voltage[fault_start:fault_end] = fault_voltage_drop + np.random.normal(0, 15, fault_duration)

    # Mark as fault
    labels[fault_start:fault_end] = 1

# Create DataFrame
df = pd.DataFrame({
    'timestamp': time,
    'current_phase_A': normal_current,
    'voltage_phase_A': normal_voltage,
    'is_fault': labels
})

# Add some additional "normal" features
df['power'] = df['current_phase_A'] * df['voltage_phase_A']
df['impedance'] = df['voltage_phase_A'] / (df['current_phase_A'] + 0.001)  # Avoid division by zero

# Save to CSV
csv_path = 'data/power_line_data.csv'
df.to_csv(csv_path, index=False)

print(f"\n✅ Dataset created successfully!")
print(f"📁 Saved to: {csv_path}")
print(f"📊 Dataset shape: {df.shape}")
print(f"⏱️  Time range: {df['timestamp'].min()} to {df['timestamp'].max()} seconds")
print(f"🔴 Fault samples: {df['is_fault'].sum()} ({df['is_fault'].sum() / len(df) * 100:.2f}%)")
print(f"🟢 Normal samples: {(df['is_fault'] == 0).sum()} ({(df['is_fault'] == 0).sum() / len(df) * 100:.2f}%)")

# Show sample of data
print("\n📋 Sample data (first 10 rows):")
print(df.head(10))