import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

def simulate_ultrasound_with_prediction(output_folder="radar_plots", dropout_interval=None):
    """
    dropout_interval: tuple (start_time, end_time) in seconds. 
                      e.g., (4.5, 5.5) will null out data between these times.
    """
    # --- Configuration ---
    c = 343.0
    max_range = 3.0
    duration = 10.0
    fps = 50
    
    # Grid setup
    slow_time = np.linspace(0, duration, int(duration * fps))
    num_pings = len(slow_time)
    range_bins = 512
    
    radar_data = np.zeros((range_bins, num_pings))

    # --- Generate Trajectory ---
    stop_start = 3.5
    stop_end = 6.5
    
    raw_distances = []
    for t in slow_time:
        if t < stop_start:
            ratio = t / stop_start
            d = (1 - ratio) * 2.5 + ratio * 0.5
        elif stop_start <= t <= stop_end:
            d = 0.5
        else:
            ratio = (t - stop_end) / (duration - stop_end)
            d = (1 - ratio) * 0.5 + ratio * 2.5
        raw_distances.append(d)
    
    # Smooth path
    raw_distances = np.array(raw_distances)
    smooth_distances = gaussian_filter1d(raw_distances, sigma=fps*0.5) 
    breathing = 0.005 * np.sin(2 * np.pi * 0.3 * slow_time)
    smooth_distances += breathing

    # --- Generate Radar Data ---
    for i, target_dist in enumerate(smooth_distances):
        idx = int((target_dist / max_range) * range_bins)
        
        # Noise
        noise = np.random.gamma(shape=1.0, scale=0.1, size=range_bins)
        radar_data[:, i] += noise
        
        # Target Signal
        if 0 <= idx < range_bins:
            width = 12
            start = max(0, idx - width)
            end = min(range_bins, idx + width)
            x = np.arange(start, end)
            signal = 3.0 * np.exp(-0.5 * ((x - idx) / (width/3))**2)
            radar_data[start:end, i] += signal

    # Add Static Clutter
    clutter_ranges = [0.8, 1.8, 2.7] 
    for cr in clutter_ranges:
        c_idx = int((cr / max_range) * range_bins)
        clutter_signal = np.random.normal(0.5, 0.1, num_pings) 
        for offset in [-1, 0, 1]:
            if 0 <= c_idx + offset < range_bins:
                radar_data[c_idx+offset, :] += clutter_signal

    # --- SIMULATE DATA LOSS ---
    if dropout_interval is not None:
        drop_start_t, drop_end_t = dropout_interval
        
        # Convert time to index
        idx_start = int(drop_start_t * fps)
        idx_end = int(drop_end_t * fps)
        
        # Clamp indices
        idx_start = max(0, idx_start)
        idx_end = min(num_pings, idx_end)
        
        # Null out the data
        radar_data[:, idx_start:idx_end] = 0.0
        print(f"Simulating data loss from {drop_start_t}s to {drop_end_t}s")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.imshow(radar_data, aspect='auto', origin='upper', cmap='viridis', 
               extent=[0, duration, max_range, 0], vmin=0, vmax=3.5)

    plt.title("Pulse compressed image (Simulated Data Loss)", fontsize=10)
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Range (meters)", fontsize=10)

    plt.tick_params(axis='both', which='major', labelsize=14)

    cbar = plt.colorbar()
    cbar.set_label("Amplitude", fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # --- Overlays ---
    # 1. Ground Truth (Red - Always solid)
    gt_y = 0.15
    plt.plot([stop_start, stop_end], [gt_y, gt_y], 
             color='red', linewidth=4, label='Ground Truth: NEAR')
    plt.text((stop_start + stop_end)/2, gt_y - 0.05, "GT", 
             color='red', fontweight='bold', ha='center', va='bottom', fontsize=10)

    # 2. Model Prediction (White - Broken if dropped out)
    pred_start = stop_start
    pred_end = stop_end
    pred_y = 0.40 
    
    # Logic to handle the split line
    if dropout_interval:
        d_start, d_end = dropout_interval
        
        # We define two potential segments: Before drop and After drop
        # Segment 1: Start -> Drop Start (if valid)
        seg1_end = min(pred_end, d_start)
        if seg1_end > pred_start:
            plt.plot([pred_start, seg1_end], [pred_y, pred_y], 
                     color='white', linewidth=4, label='Model prediction: NEAR')
            # Place the text on the first segment
            plt.text((pred_start + seg1_end)/2, pred_y - 0.05, "PRED", 
                     color='white', fontweight='bold', ha='center', va='bottom', fontsize=10)
            
            # Label has been used, don't reuse it for second segment
            label_second = None 
        else:
            label_second = 'Model prediction: NEAR'

        # Segment 2: Drop End -> End (if valid)
        seg2_start = max(pred_start, d_end)
        if seg2_start < pred_end:
            plt.plot([seg2_start, pred_end], [pred_y, pred_y], 
                     color='white', linewidth=4, label=label_second)
            
            # If segment 1 didn't exist, put text here
            if seg1_end <= pred_start:
                plt.text((seg2_start + pred_end)/2, pred_y - 0.05, "PRED", 
                         color='white', fontweight='bold', ha='center', va='bottom', fontsize=10)

    else:
        # Standard continuous line
        plt.plot([pred_start, pred_end], [pred_y, pred_y], 
                 color='white', linewidth=4, label='Model prediction: NEAR')
        plt.text((pred_start + pred_end)/2, pred_y - 0.05, "PRED", 
                 color='white', fontweight='bold', ha='center', va='bottom', fontsize=10)

    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # --- SAVE LOGIC ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    
    if dropout_interval:
        filename = "ultrasound_simulation_prediction_with_dropout.png"
    else:
        filename = "ultrasound_simulation_prediction_100performance.png"
        
    file_path = os.path.join(output_folder, filename)
    
    plt.savefig(file_path, dpi=300)
    print(f"Figure saved to: {file_path}")

    plt.show()

if __name__ == "__main__":
    # Simulating data loss between 4.5s and 5.5s
    # The white prediction line will now be broken during this gap
    simulate_ultrasound_with_prediction(
        output_folder="/home/yslcoat/projects/medium_articles/autoreview/figures",
        dropout_interval=(4.5, 5.5) 
    )