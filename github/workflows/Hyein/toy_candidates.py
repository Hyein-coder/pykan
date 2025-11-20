from kan.experiments.multkan_hparam_sweep import sweep_multkan, evaluate_params
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-1, 1, 30)
x2_grid = np.linspace(-1, 1, 30)
# x3_grid = np.linspace(-1, 1, 10)
# x1, x2, x3 = np.meshgrid(x1_grid, x2_grid, x3_grid)
# X = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
# y = np.exp(-x1) + x2 - x3**2
# y = 5 * np.exp(np.sin(x1)) + 3 * x2 - x3

x1, x2= np.meshgrid(x1_grid, x2_grid)
# X = np.stack((x1.flatten(), x2.flatten()), axis=1)
#%%
import os
save_dir = os.path.join(os.getcwd(), "github\workflows\Hyein\example_toys")
y = x1**2 + x2/4
eqn = "x1^2+0.25x2"

fig = plt.figure(figsize=(10, 8))

# Single variable
# ax = fig.add_subplot(111)
# ax.plot(x1_grid, y)
# ax.set_xlabel('x0')
# ax.set_ylabel('y')
# plt.savefig(os.path.join(save_dir, f"{eqn}.png"))
# plt.show()

# Double variable
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x1, x2, y, cmap='viridis', edgecolor='none')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('y')
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.savefig(os.path.join(save_dir, f"{eqn}.png"))
plt.show()

