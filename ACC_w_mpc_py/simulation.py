import numpy as np
import matplotlib.pyplot as plt
from vehicle_dynamics import EgoVehicle, LeadVehicle
from mpc_controller import MPCController
from performance_analysis import plot_performance

def run_simulation(ego_vehicle, lead_vehicle, mpc_controller, dt, simulation_time):
    """
    Run the ACC simulation
    
    Parameters:
    -----------
    ego_vehicle : EgoVehicle
        Ego vehicle instance
    lead_vehicle : LeadVehicle
        Lead vehicle instance
    mpc_controller : MPCController
        MPC controller instance
    dt : float
        Time step (s)
    simulation_time : float
        Total simulation time (s)
        
    Returns:
    --------
    tuple
        Time history and simulation results
    """
    print("Starting simulation...")
    
    # Storage for plotting
    time_history = []
    ego_pos_history = []
    ego_vel_history = []
    ego_accel_history = []
    lead_pos_history = []
    lead_vel_history = []
    distance_history = []
    gap_history = []  # Store current gap setting
    
    # Simulation loop
    current_time = 0.0
    while current_time < simulation_time:
        # Change gap setting every 20 seconds
        if current_time < 20.0:
            mpc_controller.set_gap(1)  # Shortest gap
        elif current_time < 40.0:
            mpc_controller.set_gap(2)  # Medium gap
        else:
            mpc_controller.set_gap(3)  # Longest gap
            
        # Store current states
        time_history.append(current_time)
        ego_pos_history.append(ego_vehicle.position)
        ego_vel_history.append(ego_vehicle.velocity)
        ego_accel_history.append(ego_vehicle.acceleration)
        lead_pos_history.append(lead_vehicle.position)
        lead_vel_history.append(lead_vehicle.velocity)
        distance_history.append(lead_vehicle.position - ego_vehicle.position)
        gap_history.append(mpc_controller.gap_settings[mpc_controller.current_gap])
        
        # Get control input from MPC
        ego_vehicle.acceleration = mpc_controller.calculate_control_input(
            ego_vehicle, lead_vehicle, dt)
        
        # Update vehicle states
        ego_vehicle.update_state(dt)
        lead_vehicle.update_state(dt, current_time)
        
        current_time += dt
        
        # Print progress every 5 seconds
        if int(current_time) % 5 == 0 and current_time > 0:
            print(f"Simulation time: {current_time:.1f}s")
    
    print("Simulation completed. Preparing plots...")
    return (time_history, ego_pos_history, ego_vel_history, ego_accel_history,
            lead_pos_history, lead_vel_history, distance_history, gap_history)

def main():
    """Main function to run the simulation"""
    print("Initializing simulation parameters...")
    
    # Simulation parameters
    dt = 0.1
    simulation_time = 60.0  # 60 seconds to test all gap settings
    
    # Convert desired speed from km/h to m/s
    desired_speed = 55.0 * 1000 / 3600  # 55 km/h to m/s
    
    print("Creating vehicle instances...")
    # Create vehicle instances
    ego_vehicle = EgoVehicle(
        initial_position=0.0,
        initial_velocity=desired_speed,  # Start at desired speed
        max_acceleration=2.0,
        min_acceleration=-3.0,
        max_velocity=desired_speed * 1.2  # Allow some overshoot
    )
    
    # Set initial lead vehicle position with safe distance
    initial_distance = 30.0  # meters
    lead_vehicle = LeadVehicle(
        initial_position=initial_distance,
        initial_velocity=0.0  # Will be updated by the sinusoidal function
    )
    
    print("Creating MPC controller...")
    # Create MPC controller with SLSQP method
    mpc_controller = MPCController(
        prediction_horizon=10,
        control_horizon=10,
        desired_velocity=desired_speed,
        desired_distance=20.0,
        min_safe_distance=10.0,  # Increased minimum safe distance
        q_velocity=1.0,
        q_distance=2.0,  # Increased weight for distance tracking
        r_acceleration=0.1,
        r_jerk=0.1,
        max_jerk=2.0,
        opt_method='SLSQP'
    )
    
    print("Running simulation...")
    # Run simulation
    results = run_simulation(ego_vehicle, lead_vehicle, mpc_controller, dt, simulation_time)
    
    print("Generating plots...")
    # Plot results and performance metrics
    plot_performance(
        results[0],  # time_history
        results[1],  # ego_pos_history
        results[2],  # ego_vel_history
        results[3],  # ego_accel_history
        results[4],  # lead_pos_history
        results[5],  # lead_vel_history
        results[6],  # distance_history
        mpc_controller.desired_velocity,
        mpc_controller.desired_distance
    )
    print("Simulation complete!")

if __name__ == "__main__":
    main() 