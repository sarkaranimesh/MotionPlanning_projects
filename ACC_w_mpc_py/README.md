# Adaptive Cruise Control with Model Predictive Control

This project implements an Adaptive Cruise Control (ACC) system using Model Predictive Control (MPC). The system maintains a safe distance from the leading vehicle while following a desired speed profile.

## Project Structure

- `vehicle_model.py`: Contains the vehicle dynamics model
- `mpc_controller.py`: Implements the MPC controller
- `simulation.py`: Main simulation environment
- `utils.py`: Helper functions and utilities
- `visualization.py`: Plotting and visualization tools

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python simulation.py
```

## Features

- Vehicle dynamics modeling
- MPC-based control strategy
- Safe distance maintenance
- Speed profile tracking
- Real-time visualization 