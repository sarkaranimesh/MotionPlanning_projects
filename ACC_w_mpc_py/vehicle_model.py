import numpy as np

class VehicleModel:
    def __init__(self, dt=0.1):
        """
        Initialize the vehicle model
        
        Parameters:
        -----------
        dt : float
            Time step for simulation (default: 0.1s)
        """
        self.dt = dt
        
        # Vehicle parameters
        self.mass = 1500  # kg
        self.drag_coef = 0.3
        self.rolling_resistance = 0.01
        self.max_accel = 2.0  # m/s^2
        self.max_decel = -3.0  # m/s^2
        
    def update(self, state, control):
        """
        Update vehicle state based on current state and control input
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state [position, velocity]
        control : float
            Control input (acceleration)
            
        Returns:
        --------
        numpy.ndarray
            Updated state
        """
        pos, vel = state
        
        # Clamp control input
        control = np.clip(control, self.max_decel, self.max_accel)
        
        # Calculate forces
        drag_force = self.drag_coef * vel**2
        rolling_force = self.rolling_resistance * self.mass * 9.81
        
        # Calculate acceleration
        accel = control - (drag_force + rolling_force) / self.mass
        
        # Update state
        new_vel = vel + accel * self.dt
        new_pos = pos + vel * self.dt + 0.5 * accel * self.dt**2
        
        return np.array([new_pos, new_vel])
    
    def get_safe_distance(self, velocity):
        """
        Calculate safe following distance based on velocity
        
        Parameters:
        -----------
        velocity : float
            Current velocity of the vehicle
            
        Returns:
        --------
        float
            Safe following distance
        """
        # Time gap of 2 seconds plus minimum distance of 5 meters
        return max(5.0, 2.0 * velocity) 