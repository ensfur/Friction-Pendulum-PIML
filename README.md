# Friction-Pendulum-PIML
# Physics-Informed Machine Learning for Damped Pendulum

This repository contains an implementation of a Physics-Informed Neural Network (PINN) model to simulate and learn the behavior of a damped pendulum.

![Animation](Friction_Pendulum_Animation.gif)

---

## 📌 Problem Description

We aim to solve the second-order nonlinear ODE of a frictional pendulum:

\[
\theta'' + b\theta' + \frac{g}{l} \sin(\theta) = 0
\]

with initial conditions:

- θ(0) = 1.0 rad
- θ'(0) = 0.0 rad/s

---

## 🔍 Features

- Physics-informed loss incorporating the differential equation
- Initial condition loss
- Comparison with analytical linear damping solution
- Animated simulation of pendulum motion
- No external dataset required

---

## 🧠 Technologies Used

- Python 3
- PyTorch
- NumPy
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/damped-pendulum-pinn.git
    cd damped-pendulum-pinn
    ```

2. Install dependencies:
    ```bash
    pip install torch numpy matplotlib
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

4. Check:
    - `Result of Friction Pendulum.png` for prediction vs ground truth
    - `Friction_Pendulum_Animation.gif` for pendulum motion

---

## 📈 Results

- The model learns the oscillatory behavior of the pendulum.
- The prediction closely matches the analytical solution for small angles.

---

## 🧩 Future Work

- Extend to double pendulum
- Include external forces (driven pendulum)
- Compare different network architectures

---

## 🤝 License

This project is open source and available under the MIT License.

