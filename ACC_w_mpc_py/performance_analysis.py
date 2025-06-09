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
                    speed_history=None):
    """
    Plot performance metrics for the ACC controller
    
    Parameters:
    -----------
    time_history : list
        Time points
    ego_pos_history : list
        Ego vehicle position history
    ego_vel_history : list
        Ego vehicle velocity history
    lead_pos_history : list
        Lead vehicle position history
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
    gap_history : list, optional
        History of time gap settings
    weight_history : list, optional
        History of MPC weights
    speed_history : list, optional
        History of desired speed changes
    """
    # Calculate jerk
    dt = time_history[1] - time_history[0]
    jerk = np.diff(ego_accel_history) / dt
    jerk_time = time_history[:-1]  # One less point for jerk
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(8, 2)  # Added one more row for speed changes
    
    # Position tracking
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time_history, ego_pos_history, label='Ego Vehicle Position')
    ax0.plot(time_history, lead_pos_history, label='Lead Vehicle Position')
    ax0.set_ylabel('Position (m)')
    ax0.legend()
    ax0.grid(True)
    ax0.set_title('Vehicle Positions')
    
    # Velocity tracking
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(time_history, ego_vel_history, label='Ego Vehicle')
    ax1.plot(time_history, lead_vel_history, label='Lead Vehicle')
    if speed_history is not None:
        ax1.plot(time_history, speed_history, 'r--', label='Desired Speed')
    else:
        ax1.axhline(y=desired_velocity, color='r', linestyle='--', label='Desired Speed')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Velocity Tracking')
    
    # Distance tracking
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(time_history, distance_history, label='Actual Distance')
    if gap_history is not None:
        # Plot desired distance based on current velocity and gap setting
        desired_distances = [v * g for v, g in zip(ego_vel_history, gap_history)]
        ax2.plot(time_history, desired_distances, 'r--', label='Desired Distance')
    else:
        ax2.axhline(y=desired_distance, color='r', linestyle='--', label='Desired Distance')
    ax2.set_ylabel('Distance (m)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Distance Tracking')
    
    # Gap settings
    if gap_history is not None:
        ax3 = fig.add_subplot(gs[3, :])
        ax3.plot(time_history, gap_history, 'g-', label='Time Gap')
        ax3.set_ylabel('Time Gap (s)')
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('Time Gap Settings')
    
    # Weight changes
    if weight_history is not None:
        ax4 = fig.add_subplot(gs[4, :])
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
        
        for key, values in weights.items():
            ax4.plot(time_history, values, label=key)
        ax4.set_ylabel('Weight Value')
        ax4.legend()
        ax4.grid(True)
        ax4.set_title('MPC Weight Changes')
    
    # Speed changes
    if speed_history is not None:
        ax5 = fig.add_subplot(gs[5, :])
        ax5.plot(time_history, speed_history, 'b-', label='Desired Speed')
        ax5.set_ylabel('Speed (m/s)')
        ax5.legend()
        ax5.grid(True)
        ax5.set_title('Desired Speed Changes')
    
    # Acceleration
    ax6 = fig.add_subplot(gs[6, 0])
    ax6.plot(time_history, ego_accel_history)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.grid(True)
    ax6.set_title('Acceleration')
    
    # Jerk
    ax7 = fig.add_subplot(gs[6, 1])
    ax7.plot(jerk_time, jerk)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Jerk (m/s³)')
    ax7.grid(True)
    ax7.set_title('Jerk')
    
    # Error distributions
    ax8 = fig.add_subplot(gs[7, 0])
    vel_error = np.array(ego_vel_history) - desired_velocity
    ax8.hist(vel_error, bins=30, density=True)
    ax8.set_xlabel('Velocity Error (m/s)')
    ax8.set_ylabel('Density')
    ax8.grid(True)
    ax8.set_title('Velocity Error Distribution')
    
    ax9 = fig.add_subplot(gs[7, 1])
    if gap_history is not None:
        dist_error = np.array(distance_history) - np.array(desired_distances)
    else:
        dist_error = np.array(distance_history) - desired_distance
    ax9.hist(dist_error, bins=30, density=True)
    ax9.set_xlabel('Distance Error (m)')
    ax9.set_ylabel('Density')
    ax9.grid(True)
    ax9.set_title('Distance Error Distribution')
    
    plt.tight_layout()
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