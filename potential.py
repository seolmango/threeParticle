import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams['axes.unicode_minus'] = False

ALHPA = 0.012174

def potential(x, y):
    return -(1 - ALHPA) / math.sqrt((x-ALHPA)**2 + y**2) - ALHPA / math.sqrt((x+1-ALHPA)**2 + y**2) - (x ** 2 + y ** 2) / 2

print(potential(-1, 0))
size = 100
range_scale = 1.5
x = np.linspace(-range_scale, range_scale, size)
y = np.linspace(-range_scale, range_scale, size)

X, Y = np.meshgrid(x, y)
Z = []
for i in range(size):
    Z.append([])
    for j in range(size):
        Z[i].append(max(potential(X[i][j], Y[i][j]), -2.25))
Z = np.array(Z)
print(Z.max())
print(Z.min())

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
# 3D
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')

# Contour
# ax = fig.add_subplot(111)
# contour_line = ax.contour(X, Y, Z, levels=8, colors='k', linewidths=1, linestyles='--')
# contour_color = ax.contourf(X, Y, Z, levels=256, cmap='jet')

# ax.clabel(contour_line, contour_line.levels, inline=True, fontsize=8)
# fig.colorbar(contour_color, ax=ax, shrink=0.5)
# ax.set_aspect('equal')
plt.show()