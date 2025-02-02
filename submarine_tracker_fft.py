import numpy as np

data_path = '/Users/chenab/Downloads/HW1 Subdata (1).npy'

d = np.load(data_path) # huge matrix of size 262144 x 49 (columns contain flattened 3d matrix of size 64x64x64)
print(np.shape(d))

import matplotlib.pyplot as plt
# import libraries for plotting isosurfaces
import plotly
import plotly.graph_objs as go
# utility for clearing output of cell as loop runs in notebook
from IPython.display import clear_output



Lh = 10; # length of spatial domain (cube of side L = 2*10).
N_grid = 64; # number of grid points/Fourier modes in each direction
xx = np.linspace(-Lh, Lh, N_grid+1) #spatial grid in x dir
x = xx[0:N_grid]
y = x # same grid in y,z direction
z = x

K_grid = (2*np.pi/(2*Lh))*np.linspace(-N_grid/2, N_grid/2 -1, N_grid) # frequency grid for one coordinate

xv, yv, zv = np.meshgrid( x, y, z) # generate 3D meshgrid for plotting

k_vals = (2*np.pi/(2*Lh)) * np.linspace(-N_grid/2, N_grid/2-1, N_grid)
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')

from scipy.fftpack import fftn, ifftn, fftshift
signal_array = []
for j in range(0, 49, 1):
    signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    signal_array.append(signal)
    
signal_array = np.array(signal_array)  
print(np.shape(signal_array))

fft_signal_array = []
for j in range(0, 49, 1):
    fft_signal = fftn(signal_array[j])
    fft_signal_shifted = fftshift(fft_signal)
    fft_signal_array.append(fft_signal_shifted)
    
fft_signal_array = np.array(fft_signal_array)

fft_signal_avg = np.zeros_like(fft_signal_array[0])
for j in range(0, 49, 1):
    fft_signal_avg += fft_signal_array[j]

fft_signal_avg /= len(fft_signal_array)

print(fft_signal_avg.shape)

abs_avg = np.abs(fft_signal_avg)
max_index = np.unravel_index(np.argmax(abs_avg), abs_avg.shape)
kx_center = kx[max_index]
ky_center = ky[max_index]
kz_center = kz[max_index]

print("Dominant frequency indices:", max_index)
print("Center frequency (kx, ky, kz) = ", kx_center, ky_center, kz_center)


abs_avg_norm = abs_avg / np.max(abs_avg)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

im1 = ax1.imshow(abs_avg_norm[:, :, max_index[2]].T, 
                 extent=[k_vals[0], k_vals[-1], k_vals[0], k_vals[-1]], 
                 origin='lower')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_title(f'kx vs ky slice at kz = -6.91')



im2 = ax2.imshow(abs_avg_norm[:, max_index[1], :].T,
                 extent=[k_vals[0], k_vals[-1], k_vals[0], k_vals[-1]], 
                 origin='lower')
ax2.set_xlabel('kx')
ax2.set_ylabel('kz')
ax2.set_title(f'kx vs kz slice at ky = 5.34')



im3 = ax3.imshow(abs_avg_norm[max_index[0], :, :].T,
                 extent=[k_vals[0], k_vals[-1], k_vals[0], k_vals[-1]], 
                 origin='lower')
ax3.set_xlabel('ky')
ax3.set_ylabel('kz')
ax3.set_title(f'ky vs kz slice at kx = 2.199')


plt.colorbar(im3)

plt.tight_layout()
plt.show()

from scipy.fftpack import ifftshift

var = 0.5
tau = 1.0 / (2*var)
gaussian_filter = np.exp(-tau * ((kx - kx_center)**2 + (ky - ky_center)**2 + (kz - kz_center)**2))
print(np.shape(fft_signal_array))
filtered_signal_array = []
for j in range(0, 49, 1):
    fft_signal_shifted = fft_signal_array[j]
    fft_signal_filtered = gaussian_filter * fft_signal_shifted
    
    F3_filtered = ifftshift(fft_signal_filtered)
    filtered_space = ifftn(F3_filtered)
    
    filtered_signal_array.append(filtered_space)
filtered_signal_array = np.array(filtered_signal_array)

submarine_positions = []

for j in range(0, 49, 1):
    filtered_spatial_signal = np.abs(filtered_signal_array[j])
    max_index = np.unravel_index(np.argmax(filtered_spatial_signal), filtered_spatial_signal.shape)
    x_max_index, y_max_index, z_max_index = max_index
    x_sub = x[x_max_index]
    y_sub = y[y_max_index]
    z_sub = z[z_max_index]
    submarine_positions.append([x_sub, y_sub, z_sub])
submarine_positions = np.array(submarine_positions)

print(submarine_positions)

#Plotting in 3D
fig = plt.figure()
ax = plt.axes(projection='3d')  
scatter = ax.scatter(submarine_positions[:, 0], submarine_positions[:, 1], submarine_positions[:, 2], c=time_array, cmap='viridis')
ax.plot(submarine_positions[:, 0], submarine_positions[:, 1], submarine_positions[:, 2], 'k-', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Denoised Submarine path in 3D space')
cbar = plt.colorbar(scatter, shrink=0.6, pad=0.2)
cbar.set_label('Time (hours)')
plt.show()

#Plotting in XY Plane
time_array = 0.5 * np.arange(0, 49, 1)  # in hours, since each index is after 30 minute interval

plt.figure(figsize=(10,6))
plt.scatter(submarine_positions[:,0], submarine_positions[:,1], c=time_array, cmap='viridis')
plt.plot(submarine_positions[:,0], submarine_positions[:,1], 'k-', alpha=0.5)
cbar = plt.colorbar(shrink=0.6, pad=0.2)
cbar.set_label('Time (hours)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Submarine path in the X-Y plane')
plt.grid(True)
plt.show()


