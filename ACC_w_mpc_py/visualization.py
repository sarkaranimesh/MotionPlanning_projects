import pygame
import numpy as np
import sys

class VehicleVisualizer:
    def __init__(self, width=1200, height=400, scale=10):
        """
        Initialize the visualization
        
        Parameters:
        -----------
        width : int
            Window width in pixels
        height : int
            Window height in pixels
        scale : float
            Pixels per meter for visualization
        """
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ACC Simulation Visualization")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (200, 200, 200)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        
        # Vehicle dimensions
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 1.8   # meters
        
    def draw_road(self):
        """Draw the road and lane markings"""
        # Draw road
        pygame.draw.rect(self.screen, self.GRAY, (0, self.height//2 - 50, self.width, 100))
        
        # Draw lane markings
        dash_length = 20
        gap_length = 20
        x = 0
        while x < self.width:
            pygame.draw.rect(self.screen, self.WHITE, (x, self.height//2 - 2, dash_length, 4))
            x += dash_length + gap_length
    
    def draw_vehicle(self, x, y, color, velocity, acceleration):
        """
        Draw a vehicle at the specified position
        
        Parameters:
        -----------
        x : float
            Position in meters
        y : float
            Vertical position in pixels
        color : tuple
            RGB color tuple
        velocity : float
            Vehicle velocity in m/s
        acceleration : float
            Vehicle acceleration in m/s²
        """
        # Convert position to pixels
        x_pixels = int(x * self.scale)
        
        # Draw vehicle body
        vehicle_rect = pygame.Rect(
            x_pixels - int(self.vehicle_length * self.scale / 2),
            y - int(self.vehicle_width * self.scale / 2),
            int(self.vehicle_length * self.scale),
            int(self.vehicle_width * self.scale)
        )
        pygame.draw.rect(self.screen, color, vehicle_rect)
        
        # Convert velocity to km/h for display
        velocity_kmh = velocity * 3.6
        
        # Draw velocity and acceleration text
        vel_text = self.font.render(f"v: {velocity_kmh:.1f} km/h", True, self.BLACK)
        acc_text = self.font.render(f"a: {acceleration:.1f} m/s²", True, self.BLACK)
        self.screen.blit(vel_text, (x_pixels - 50, y - 40))
        self.screen.blit(acc_text, (x_pixels - 50, y + 20))
    
    def draw_distance(self, ego_x, lead_x, y):
        """
        Draw the distance between vehicles
        
        Parameters:
        -----------
        ego_x : float
            Ego vehicle position in meters
        lead_x : float
            Lead vehicle position in meters
        y : int
            Vertical position in pixels
        """
        distance = lead_x - ego_x
        x_pixels = int(ego_x * self.scale)
        
        # Draw distance line
        pygame.draw.line(self.screen, self.BLACK,
                        (x_pixels, y),
                        (x_pixels + int(distance * self.scale), y),
                        2)
        
        # Draw distance text
        dist_text = self.font.render(f"d: {distance:.1f} m", True, self.BLACK)
        self.screen.blit(dist_text, (x_pixels + int(distance * self.scale / 2) - 40, y - 30))
    
    def update(self, ego_pos, ego_vel, ego_accel, lead_pos, lead_vel, lead_accel):
        """
        Update the visualization with new vehicle states
        
        Parameters:
        -----------
        ego_pos : float
            Ego vehicle position in meters
        ego_vel : float
            Ego vehicle velocity in m/s
        ego_accel : float
            Ego vehicle acceleration in m/s²
        lead_pos : float
            Lead vehicle position in meters
        lead_vel : float
            Lead vehicle velocity in m/s
        lead_accel : float
            Lead vehicle acceleration in m/s²
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw road
        self.draw_road()
        
        # Draw vehicles
        self.draw_vehicle(ego_pos, self.height//2, self.BLUE, ego_vel, ego_accel)
        self.draw_vehicle(lead_pos, self.height//2, self.RED, lead_vel, lead_accel)
        
        # Draw distance
        self.draw_distance(ego_pos, lead_pos, self.height//2 + 60)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        pygame.time.Clock().tick(30) 