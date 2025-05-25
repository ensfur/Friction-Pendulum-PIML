# Physics-Informed Neural Network for Damped Pendulum

This project implements a Physics-Informed Neural Network (PINN) to model the behavior of a damped pendulum. The PINN leverages the governing differential equation of the system to inform the training process, ensuring that the model's predictions adhere to known physical laws.

![Damped Pendulum Animation](Damped_Pendulum_Animation.gif)

## 🧠 Overview

The damped pendulum is described by the second-order nonlinear ordinary differential equation (ODE):

\[
\theta''(t) + b\theta'(t) + \frac{g}{l} \sin(\theta(t)) = 0
\]

Where:
- \( \theta(t) \): Angular displacement at time \( t \)
- \( b \): Damping coefficient
- \( g \): Acceleration due to gravity
- \( l \): Length of the pendulum

The PINN is trained to minimize a loss function that combines:
- The residual of the ODE (physics-informed loss)
- The discrepancy from initial conditions

## 📁 Project Structure

- `main.py`: Contains the implementation of the PINN and training loop.
- `Result_of_Damped_Pendulum.png`: Plot comparing the PINN prediction with the analytical solution.
- `Damped_Pendulum_Animation.gif`: Animation visualizing the pendulum's motion over time.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `torch`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy torch matplotlib
