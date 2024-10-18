import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Array size (given)
M = 918855  # x-axis (rows)
N = 313714  # y-axis (columns)

# Function to calculate total number of chunks based on chunk sizes
def total_chunks(chunk_sizes):
    x_chunk, y_chunk = chunk_sizes
    chunks_x = np.ceil(M / x_chunk)
    chunks_y = np.ceil(N / y_chunk)
    return chunks_x * chunks_y

# Define a finer grid of x and y chunk sizes starting from 0 to 10,000 for the 2D array optimization
x_fine_chunk_sizes = np.linspace(1, 1000, 10)
y_fine_chunk_sizes = np.linspace(1, 1000, 10)

# Create a meshgrid for plotting
X_fine, Y_fine = np.meshgrid(x_fine_chunk_sizes, y_fine_chunk_sizes)

# Calculate the total number of chunks for each combination of chunk sizes
Z_fine = np.zeros_like(X_fine)
for i in range(X_fine.shape[0]):
    for j in range(X_fine.shape[1]):
        Z_fine[i, j] = total_chunks([X_fine[i, j], Y_fine[i, j]])

# Plot the 3D surface for the gradient curve
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf_fine = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='jet', edgecolor='none')

# Adjust axes to start from 0 and ensure chunk sizes and chunk counts grow together

ax.set_zlim(0, np.max(Z_fine))

# Add labels and color bar
ax.set_xlabel('Chunk Size (x-axis)')
ax.set_ylabel('Chunk Size (y-axis)')
ax.set_zlabel('Total Number of Chunks')
ax.set_title('Optimization Gradient Curve for 2D Array (918,855 x 313,714)')

fig.colorbar(surf_fine, shrink=0.5, aspect=5)
plt.show()
