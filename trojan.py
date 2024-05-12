import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

ALPHA = 0.000954

# Define the system of ODEs
def equations_of_motion(w, t):
    x, dxdt, y, dydt = w
    dx2dt2 = -(1-ALPHA)*(x-ALPHA)/(np.sqrt((x-ALPHA)**2 + y**2)**3) \
              - ALPHA*(x+1-ALPHA)/np.sqrt((x+1-ALPHA)**2 + y**2)**3 + 2*dydt + x
    dy2dt2 = -(1-ALPHA)*y/(np.sqrt((x-ALPHA)**2 + y**2)**3) \
              - ALPHA*y/np.sqrt((x+1-ALPHA)**2 + y**2)**3 - 2*dxdt + y
    return [dxdt, dx2dt2, dydt, dy2dt2]

def potential(x, y):
    return -(1 - ALPHA) / math.sqrt((x - ALPHA) ** 2 + y ** 2) - ALPHA / math.sqrt((x + 1 - ALPHA) ** 2 + y ** 2) - (x ** 2 + y ** 2) / 2

size = 100
range_scale = 1.5
x = np.linspace(-range_scale, range_scale, size)
y = np.linspace(-range_scale, range_scale, size)

X, Y = np.meshgrid(x, y)
Z = []
for i in range(size):
    Z.append([])
    for j in range(size):
        Z[i].append(max(potential(X[i][j], Y[i][j]), -2))
Z = np.array(Z)
print(Z.max())
print(Z.min())

# initial_conditions = [
#     [-0.509, 0.0259, 0.883, 0.0149, "1"],
#     [-0.524, 0.0647, 0.909, 0.0367, "2"],
#     [-0.524, 0.0780, 0.920, 0.0430, "3"]
# ]
initial_conditions = [
    # [-0.509, -0.0259, 0.883, -0.0149, "4"],
    [-0.532, 0.0780, 0.92, 0.0430, "5"]
]
# t = 80
fig = plt.figure(figsize=(8,8))
plt.title(f'Trojan Asteroid Trajectories')
for i, w0 in enumerate(initial_conditions):
    t = np.linspace(0, 500, 2000)
    wsol = odeint(equations_of_motion, w0[:-1], t)
    x_vals = wsol[:, 0]
    y_vals = wsol[:, 2]
    plt.plot(x_vals, y_vals, label=f'Trajectory {w0[-1]}')
# contour_color = plt.contourf(X, Y, Z, levels=256, cmap='jet')
# contour_line = plt.contour(X, Y, Z, levels=8, colors='k', linewidths=1, linestyles='--')
# plt.clabel(contour_line, contour_line.levels, inline=True, fontsize=8)
# fig.colorbar(contour_color, ax=plt.gca(), shrink=0.5)
plt.xlabel('x')
plt.ylabel('y')
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()