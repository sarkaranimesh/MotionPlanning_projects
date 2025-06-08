import numpy as np

class EgoVehicle:
    def __init__(self, initial_position=0.0, initial_velocity=0.0,
                 max_acceleration=2.0, min_acceleration=-3.0,
                 max_velocity=30.0):
        """
        Initialize the ego vehicle
        
        Parameters:
        -----------
        initial_position : float
            Initial longitudinal position (m)
        initial_velocity : float
            Initial longitudinal velocity (m/s)
        max_acceleration : float
            Maximum allowed acceleration (m/s²)
        min_acceleration : float
            Minimum allowed acceleration (m/s²)
        max_velocity : float
            Maximum allowed velocity (m/s)
        """
        self.position = initial_position
        self.velocity = initial_velocity
        self.acceleration = 0.0
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_velocity = max_velocity
        
    def update_state(self, dt):
        """
        Update vehicle state using discrete-time kinematic model
        
        Parameters:
        -----------
        dt : float
            Time step (s)
        """
        # Clamp acceleration
        self.acceleration = np.clip(self.acceleration, 
                                  self.min_acceleration, 
                                  self.max_acceleration)
        
        # Update position and velocity
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt
        
        # Clamp velocity
        self.velocity = np.clip(self.velocity, 0.0, self.max_velocity)
        
    def get_state(self):
        """Get current vehicle state"""
        return np.array([self.position, self.velocity])

class LeadVehicle:
    def __init__(self, initial_position=50.0, initial_velocity=15.0):
        """
        Initialize the lead vehicle
        
        Parameters:
        -----------
        initial_position : float
            Initial longitudinal position (m)
        initial_velocity : float
            Initial longitudinal velocity (m/s)
        """
        self.position = initial_position
        self.velocity = initial_velocity
        
    def update_state(self, dt, time):
        """
        Update lead vehicle state
        
        Parameters:
        -----------
        dt : float
            Time step (s)
        time : float
            Current simulation time (s)
        """
        # Get velocity from behavior function
        self.velocity = get_lead_vehicle_velocity(time)
        
        # Update position
        self.position += self.velocity * dt
        
    def get_state(self):
        """Get current vehicle state"""
        return np.array([self.position, self.velocity])

def get_lead_vehicle_velocity(time):
    """
    Determine lead vehicle velocity based on time
    
    Parameters:
    -----------
    time : float
        Current simulation time (s)
        
    Returns:
    --------
    float
        Lead vehicle velocity (m/s)
    """
    # Convert km/h to m/s
    max_vel = 40.0 * 1000 / 3600  # 40 km/h to m/s
    min_vel = 0.0
    
    # Create sinusoidal pattern with 20-second period
    period = 20.0  # seconds
    angular_freq = 2 * np.pi / period
    
    # Sinusoidal velocity profile
    velocity = (max_vel - min_vel) / 2 * (1 + np.sin(angular_freq * time)) + min_vel
    
    return velocity 