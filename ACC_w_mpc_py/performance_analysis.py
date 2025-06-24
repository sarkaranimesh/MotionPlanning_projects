import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_performance_metrics(time_history, ego_vel_history, lead_vel_history,
                                ego_accel_history, distance_history,
                                desired_velocity, desired_distance):
    """
    Calculate performance metrics for the ACC controller
    
    Parameters:
    -----------
    time_history : list
        Time points
    ego_vel_history : list
        Ego vehicle velocity history
    lead_vel_history : list
        Lead vehicle velocity history
    ego_accel_history : list
        Ego vehicle acceleration history
    distance_history : list
        Following distance history
    desired_velocity : float
        Desired velocity
    desired_distance : float
        Desired following distance
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Convert lists to numpy arrays
    ego_vel = np.array(ego_vel_history)
    lead_vel = np.array(lead_vel_history)
    ego_accel = np.array(ego_accel_history)
    distance = np.array(distance_history)
    
    # Calculate velocity tracking error
    vel_error = ego_vel - desired_velocity
    vel_rmse = np.sqrt(np.mean(vel_error**2))
    vel_mae = np.mean(np.abs(vel_error))
    
    # Calculate distance tracking error
    dist_error = distance - desired_distance
    dist_rmse = np.sqrt(np.mean(dist_error**2))
    dist_mae = np.mean(np.abs(dist_error))
    
    # Calculate acceleration statistics
    accel_mean = np.mean(np.abs(ego_accel))
    accel_std = np.std(ego_accel)
    accel_max = np.max(np.abs(ego_accel))
    
    # Calculate jerk
    dt = time_history[1] - time_history[0]
    jerk = np.diff(ego_accel) / dt
    jerk_mean = np.mean(np.abs(jerk))
    jerk_std = np.std(jerk)
    jerk_max = np.max(np.abs(jerk))
    
    # Calculate comfort metrics
    # Pad jerk array to match ego_accel length for comfort index calculation
    jerk_padded = np.pad(jerk, (0, 1), mode='edge')
    comfort_index = np.mean(ego_accel**2 + jerk_padded**2)
    
    return {
        'velocity_rmse': vel_rmse,
        'velocity_mae': vel_mae,
        'distance_rmse': dist_rmse,
        'distance_mae': dist_mae,
        'accel_mean': accel_mean,
        'accel_std': accel_std,
        'accel_max': accel_max,
        'jerk_mean': jerk_mean,
        'jerk_std': jerk_std,
        'jerk_max': jerk_max,
        'comfort_index': comfort_index
    }

def plot_performance(time_history, ego_pos_history, ego_vel_history, lead_pos_history,
                    lead_vel_history, ego_accel_history, distance_history,
                    desired_velocity, desired_distance, gap_history=None, weight_history=None,
                    speed_history=None, save_path=None):
    """
    Plot performance metrics for the ACC controller with dark theme
    If save_path is provided, saves the plots as PNG images instead of showing them.
    """
    # Set dark theme
    plt.style.use('dark_background')
    
    # Calculate jerk
    dt = time_history[1] - time_history[0]
    jerk = np.diff(ego_accel_history) / dt
    jerk_time = time_history[:-1]  # One less point for jerk
    
    # Create two figures
    # Figure 1: Vehicle Kinematics
    fig1 = plt.figure(figsize=(15, 10))
    gs1 = fig1.add_gridspec(4, 1)
    
    # Position tracking
    ax0 = fig1.add_subplot(gs1[0, 0])
    ax0.plot(time_history, ego_pos_history, label='Ego Vehicle', color='#00ff00')
    ax0.plot(time_history, lead_pos_history, label='Lead Vehicle', color='#ff0000')
    ax0.set_ylabel('Position (m)', color='white')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title('Vehicle Positions', color='white')
    
    # Velocity tracking
    ax1 = fig1.add_subplot(gs1[1, 0])
    ax1.plot(time_history, ego_vel_history, label='Ego Vehicle', color='#00ff00')
    ax1.plot(time_history, lead_vel_history, label='Lead Vehicle', color='#ff0000')
    if speed_history is not None:
        ax1.plot(time_history, speed_history, '--', label='Desired Speed', color='#ffff00')
    else:
        ax1.axhline(y=desired_velocity, color='#ffff00', linestyle='--', label='Desired Speed')
    ax1.set_ylabel('Velocity (m/s)', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Velocity Tracking', color='white')
    
    # Distance tracking
    ax2 = fig1.add_subplot(gs1[2, 0])
    ax2.plot(time_history, distance_history, label='Actual Distance', color='#00ffff')
    if gap_history is not None:
        desired_distances = [v * g for v, g in zip(ego_vel_history, gap_history)]
        ax2.plot(time_history, desired_distances, '--', label='Desired Distance', color='#ffff00')
    else:
        ax2.axhline(y=desired_distance, color='#ffff00', linestyle='--', label='Desired Distance')
    ax2.set_ylabel('Distance (m)', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Distance Tracking', color='white')
    
    # Acceleration and Jerk
    ax3 = fig1.add_subplot(gs1[3, 0])
    ax3.plot(time_history, ego_accel_history, label='Acceleration', color='#ff00ff')
    ax3.plot(jerk_time, jerk, label='Jerk', color='#00ffff')
    ax3.set_xlabel('Time (s)', color='white')
    ax3.set_ylabel('Acceleration (m/s²), Jerk (m/s³)', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Acceleration and Jerk', color='white')
    
    # Figure 2: Controller Performance
    fig2 = plt.figure(figsize=(15, 10))
    gs2 = fig2.add_gridspec(3, 1)
    
    # Gap settings
    if gap_history is not None:
        ax4 = fig2.add_subplot(gs2[0, 0])
        ax4.plot(time_history, gap_history, label='Time Gap', color='#00ff00')
        ax4.set_ylabel('Time Gap (s)', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Time Gap Settings', color='white')
    
    # Weight changes
    if weight_history is not None:
        ax5 = fig2.add_subplot(gs2[1, 0])
        weights = {
            'q_velocity': [],
            'q_distance': [],
            'q_close': [],
            'q_far': [],
            'r_acceleration': [],
            'r_jerk': []
        }
        for w in weight_history:
            for key in weights:
                weights[key].append(w[key])
        
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
        for (key, values), color in zip(weights.items(), colors):
            ax5.plot(time_history, values, label=key, color=color)
        ax5.set_ylabel('Weight Value', color='white')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title('MPC Weight Changes', color='white')
    
    # Speed changes
    if speed_history is not None:
        ax6 = fig2.add_subplot(gs2[2, 0])
        ax6.plot(time_history, speed_history, label='Desired Speed', color='#ffff00')
        ax6.set_xlabel('Time (s)', color='white')
        ax6.set_ylabel('Speed (m/s)', color='white')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_title('Desired Speed Changes', color='white')
    
    # Adjust layout and show plots
    fig1.tight_layout()
    fig2.tight_layout()
    if save_path:
        fig1.savefig(f"{save_path}_1.png")
        fig2.savefig(f"{save_path}_2.png")
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()
    
    # Print performance metrics
    metrics = calculate_performance_metrics(
        time_history, ego_vel_history, lead_vel_history,
        ego_accel_history, distance_history,
        desired_velocity, desired_distance
    )
    
    print("\nPerformance Metrics:")
    print(f"Velocity RMSE: {metrics['velocity_rmse']:.2f} m/s")
    print(f"Velocity MAE: {metrics['velocity_mae']:.2f} m/s")
    print(f"Distance RMSE: {metrics['distance_rmse']:.2f} m")
    print(f"Distance MAE: {metrics['distance_mae']:.2f} m")
    print(f"Mean Acceleration: {metrics['accel_mean']:.2f} m/s²")
    print(f"Max Acceleration: {metrics['accel_max']:.2f} m/s²")
    print(f"Mean Jerk: {metrics['jerk_mean']:.2f} m/s³")
    print(f"Max Jerk: {metrics['jerk_max']:.2f} m/s³")
    print(f"Comfort Index: {metrics['comfort_index']:.2f}")

    return metrics 