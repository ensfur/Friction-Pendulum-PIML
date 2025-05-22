import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81  
l = 1.0   
b = 0.2   

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, t):
        return self.net(t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

t_train = np.linspace(0, 10, 1000)[:, None]
t_tensor = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)

t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True).to(device)
theta0 = torch.tensor([[1.0]], dtype=torch.float32).to(device)
theta0_dot = torch.tensor([[0.0]], dtype=torch.float32).to(device)

def pinn_loss():
    theta = model(t_tensor)

    theta_t = torch.autograd.grad(theta, t_tensor, torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t_tensor, torch.ones_like(theta_t), create_graph=True)[0]

    # θ'' + (b/m)*θ' + (g/l)*sin(θ) = 0
    physics = theta_tt + b * theta_t + (g / l) * torch.sin(theta)
    physics_loss = torch.mean(physics ** 2)

    theta0_pred = model(t0)
    theta0_dot_pred = torch.autograd.grad(theta0_pred, t0, torch.ones_like(theta0_pred), create_graph=True)[0]
    ic_loss = (theta0_pred - theta0) ** 2 + (theta0_dot_pred - theta0_dot) ** 2

    return physics_loss + 1 * ic_loss.mean()

num_epochs = 25000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = pinn_loss()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Training completed.")

t_test = np.linspace(0, 10, 500)[:, None]
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).to(device)
with torch.no_grad():
    theta_pred = model(t_test_tensor).cpu().numpy()

def true_solution_damped(t, theta0=1.0, b=0.2, m=1.0):
    omega0 = np.sqrt(g / l)
    gamma = b / (2 * m)
    omega_d = np.sqrt(omega0**2 - gamma**2)
    A = theta0
    return A * np.exp(-gamma * t) * np.cos(omega_d * t)

theta_true = true_solution_damped(t_test.flatten())


plt.figure(figsize=(10, 6))
plt.plot(t_test, theta_true, label="Linear Damping", linestyle="dashed")
plt.plot(t_test, theta_pred, label="PIML prediction", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Degree (rad)")
plt.title("Friction Pendulum - Solution with PIML")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Result of Friction Pendulum.png")

fig, ax = plt.subplots(figsize=(5, 5))
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], 'r--', alpha=0.5)
theta_anim = theta_pred.flatten()
xdata, ydata = [], []

def init():
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    return line, trace

def update(frame):
    theta = theta_anim[frame]
    x = l * np.sin(theta)
    y = -l * np.cos(theta)

    xdata.append(x)
    ydata.append(y)

    line.set_data([0, x], [0, y])
    trace.set_data(xdata, ydata)
    return line, trace

ani = FuncAnimation(fig, update, frames=len(theta_anim), init_func=init, blit=True, interval=20)
ani.save("Friction_Pendulum_Animation.gif", writer="pillow")
plt.show()
