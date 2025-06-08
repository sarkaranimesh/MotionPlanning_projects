import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, prediction_horizon, control_horizon, desired_velocity,
                 desired_distance, min_safe_distance, q_velocity, q_distance,
                 r_acceleration, r_jerk, max_jerk, opt_method='SLSQP'):
        """
        Initialize MPC controller
        
        Parameters:
        -----------
        prediction_horizon : int
            Number of prediction steps
        control_horizon : int
            Number of control steps
        desired_velocity : float
            Desired velocity (m/s)
        desired_distance : float
            Desired following distance (m)
        min_safe_distance : float
            Minimum safe distance (m)
        q_velocity : float
            Weight for velocity tracking
        q_distance : float
            Weight for distance tracking
        r_acceleration : float
            Weight for acceleration
        r_jerk : float
            Weight for jerk
        max_jerk : float
            Maximum allowed jerk
        opt_method : str
            Optimization method to use
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.desired_velocity = desired_velocity
        self.desired_distance = desired_distance
        self.min_safe_distance = min_safe_distance
        self.q_velocity = q_velocity
        self.q_distance = q_distance
        self.r_acceleration = r_acceleration
        self.r_jerk = r_jerk
        self.max_jerk = max_jerk
        self.opt_method = opt_method
        
        # Gap settings (time gaps in seconds)
        self.gap_settings = {
            1: 0.8,  # Shortest gap
            2: 1.5,  # Medium gap
            3: 2.5   # Longest gap
        }
        self.current_gap = 2  # Default to medium gap
        
        # Initialize control sequence
        self.control_sequence = np.zeros(control_horizon)
        
    def set_gap(self, gap_number):
        """
        Set the desired time gap
        
        Parameters:
        -----------
        gap_number : int
            Gap setting (1, 2, or 3)
        """
        if gap_number not in self.gap_settings:
            raise ValueError("Gap number must be 1, 2, or 3")
        self.current_gap = gap_number
        
    def _get_desired_distance(self, ego_velocity):
        """
        Calculate desired distance based on current velocity and gap setting
        
        Parameters:
        -----------
        ego_velocity : float
            Current ego vehicle velocity (m/s)
            
        Returns:
        --------
        float
            Desired following distance (m)
        """
        time_gap = self.gap_settings[self.current_gap]
        return max(self.min_safe_distance, ego_velocity * time_gap)
        
    def _cost_function(self, control_sequence, ego_state, lead_state, dt):
        """
        Calculate cost for MPC optimization
        
        Parameters:
        -----------
        control_sequence : numpy.ndarray
            Sequence of control inputs
        ego_state : numpy.ndarray
            Current ego vehicle state [position, velocity]
        lead_state : numpy.ndarray
            Current lead vehicle state [position, velocity]
        dt : float
            Time step (s)
            
        Returns:
        --------
        float
            Total cost
        """
        total_cost = 0.0
        current_state = ego_state.copy()
        
        # Calculate desired distance based on current velocity and gap setting
        desired_distance = self._get_desired_distance(current_state[1])
        
        for i in range(self.prediction_horizon):
            # Update state
            current_state[0] += current_state[1] * dt
            current_state[1] += control_sequence[min(i, self.control_horizon-1)] * dt
            
            # Calculate distance to lead vehicle
            distance = lead_state[0] - current_state[0]
            
            # Calculate desired distance for this step
            step_desired_distance = self._get_desired_distance(current_state[1])
            
            # Velocity tracking cost
            vel_error = current_state[1] - self.desired_velocity
            total_cost += self.q_velocity * vel_error**2
            
            # Distance tracking cost with asymmetric penalties
            dist_error = distance - step_desired_distance
            if dist_error < 0:  # If too close
                total_cost += 10.0 * self.q_distance * dist_error**2  # Stronger penalty
            else:
                total_cost += self.q_distance * dist_error**2
            
            # Acceleration cost
            if i < self.control_horizon:
                total_cost += self.r_acceleration * control_sequence[i]**2
                
                # Jerk cost
                if i > 0:
                    jerk = (control_sequence[i] - control_sequence[i-1]) / dt
                    total_cost += self.r_jerk * jerk**2
                    
                    # Penalty for exceeding max jerk
                    if abs(jerk) > self.max_jerk:
                        total_cost += 1000.0 * (abs(jerk) - self.max_jerk)**2
        
        return total_cost
        
    def calculate_control_input(self, ego_vehicle, lead_vehicle, dt):
        """
        Calculate optimal control input using MPC
        
        Parameters:
        -----------
        ego_vehicle : EgoVehicle
            Ego vehicle instance
        lead_vehicle : LeadVehicle
            Lead vehicle instance
        dt : float
            Time step (s)
            
        Returns:
        --------
        float
            Optimal control input (acceleration)
        """
        # Current states
        ego_state = np.array([ego_vehicle.position, ego_vehicle.velocity])
        lead_state = np.array([lead_vehicle.position, lead_vehicle.velocity])
        
        # Bounds for control inputs
        bounds = [(-3.0, 2.0) for _ in range(self.control_horizon)]
        
        # Initial guess for control sequence
        initial_guess = np.zeros(self.control_horizon)
        
        # Optimize control sequence
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(ego_state, lead_state, dt),
            method=self.opt_method,
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        # Update control sequence
        self.control_sequence = result.x
        
        # Return first control input
        return self.control_sequence[0] 