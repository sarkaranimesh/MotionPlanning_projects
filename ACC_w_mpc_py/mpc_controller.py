import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, horizon=10, dt=0.1, min_safe_distance=10.0):
        self.horizon = horizon
        self.dt = dt
        self.min_safe_distance = min_safe_distance
        self.max_jerk = 2.0  # Maximum allowed jerk (m/s³)
        
        # Initialize gap settings
        self.gap_settings = {
            'aggressive': 1.0,    # 1.0 seconds
            'balanced': 2.0,      # 2.0 seconds
            'conservative': 3.0   # 3.0 seconds
        }
        self.current_gap = 'balanced'  # Default gap setting
        self.desired_velocity = 15.0   # Default desired velocity (m/s)
        
        # Initialize weights with more balanced values
        self.weights = {
            'q_velocity': 1.0,      # Increased from 0.5
            'q_distance': 2.0,      # Increased from 1.0
            'r_acceleration': 0.1,  # Decreased from 0.5
            'r_jerk': 0.1,         # Decreased from 0.5
            'q_close': 10.0,       # Increased from 5.0
            'q_far': 1.0,          # Increased from 0.5
            'q_safety': 100.0,     # Kept the same
            'q_jerk_violation': 100.0  # Kept the same
        }
        
        # Store last control input for warm start
        self.last_control = None
        
    def calibrate_weights(self, weights_dict):
        """
        Calibrate MPC weights for different scenarios
        
        Parameters:
        -----------
        weights_dict : dict
            Dictionary containing new weight values. Keys can be:
            - q_velocity: Weight for velocity tracking
            - q_distance: Weight for distance tracking
            - r_acceleration: Weight for acceleration
            - r_jerk: Weight for jerk
            - q_close: Weight for being too close
            - q_far: Weight for being too far
            - q_safety: Weight for safety constraint violation
            - q_jerk_violation: Weight for jerk constraint violation
        """
        for key, value in weights_dict.items():
            if key in self.weights:
                self.weights[key] = value
            else:
                print(f"Warning: Unknown weight key '{key}'")
                
    def get_current_weights(self):
        """
        Get current weight values
        
        Returns:
        --------
        dict
            Dictionary containing current weight values
        """
        return self.weights.copy()
        
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
        
    def _cost_function(self, u, ego_state, lead_state, t):
        """
        Calculate cost for MPC optimization
        
        Parameters:
        -----------
        u : numpy.ndarray
            Sequence of control inputs
        ego_state : numpy.ndarray
            Current ego vehicle state [position, velocity]
        lead_state : numpy.ndarray
            Current lead vehicle state [position, velocity]
        t : float
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
        
        for i in range(self.horizon):
            # Update state
            current_state[0] += current_state[1] * t
            current_state[1] += u[min(i, self.horizon-1)] * t
            
            # Calculate distance to lead vehicle
            distance = lead_state[0] - current_state[0]
            
            # Calculate desired distance for this step
            step_desired_distance = self._get_desired_distance(current_state[1])
            
            # Velocity tracking cost
            vel_error = current_state[1] - self.desired_velocity
            total_cost += self.weights['q_velocity'] * vel_error**2
            
            # Distance tracking cost with asymmetric penalties
            dist_error = distance - step_desired_distance
            if dist_error < 0:  # If too close
                total_cost += self.weights['q_close'] * self.weights['q_distance'] * dist_error**2
            else:
                total_cost += self.weights['q_far'] * self.weights['q_distance'] * dist_error**2
            
            # Safety distance constraint violation
            if distance < self.min_safe_distance:
                total_cost += self.weights['q_safety'] * (self.min_safe_distance - distance)**2
            
            # Acceleration cost
            if i < self.horizon:
                total_cost += self.weights['r_acceleration'] * u[i]**2
                
                # Jerk cost
                if i > 0:
                    jerk = (u[i] - u[i-1]) / t
                    total_cost += self.weights['r_jerk'] * jerk**2
                    
                    # Penalty for exceeding max jerk
                    if abs(jerk) > self.max_jerk:
                        total_cost += self.weights['q_jerk_violation'] * (abs(jerk) - self.max_jerk)**2
        
        return total_cost
        
    def optimize(self, ego_state, lead_state, t):
        # Initial guess for control sequence (warm start)
        if self.last_control is not None:
            u0 = self.last_control
        else:
            u0 = np.zeros(self.horizon)
        
        # Bounds for control inputs
        bounds = [(-3.0, 3.0) for _ in range(self.horizon)]  # Reduced from (-5.0, 5.0)
        
        # Constraints
        constraints = []
        
        # Optimize
        result = minimize(
            self._cost_function,
            u0,
            args=(ego_state, lead_state, t),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 100,  # Increased from default
                'ftol': 1e-4,    # Added tolerance
                'disp': False
            }
        )
        
        # Store solution for warm start
        if result.success:
            self.last_control = result.x
        else:
            # If optimization fails, use last successful control or zero
            if self.last_control is not None:
                self.last_control = np.roll(self.last_control, -1)
                self.last_control[-1] = 0.0
            else:
                self.last_control = np.zeros(self.horizon)
        
        return result.x[0] if result.success else 0.0 

    def set_gap(self, gap_value):
        """
        Set the time gap for following distance
        
        Parameters:
        -----------
        gap_value : float
            Time gap in seconds (1.0 for aggressive, 2.0 for balanced, 3.0 for conservative)
        """
        # Find the closest matching gap setting
        if gap_value <= 1.0:
            self.current_gap = 'aggressive'
        elif gap_value <= 2.0:
            self.current_gap = 'balanced'
        else:
            self.current_gap = 'conservative' 