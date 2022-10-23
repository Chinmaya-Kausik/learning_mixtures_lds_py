import csv
import matplotlib.pyplot as plt
import numpy as np
from subspace_est import subspace_estimation
from clustering import clustering_fast

"""Preprocessing"""

# Initialize d, K, M, T
d = 12
K = 2
T = 400
M=12

# Open the files    
file1 = open('jog.csv')
file2 = open('wlk_sub_23.csv')

# Initialize the readers and skip the headers
jog_reader = csv.reader(file1)
walk_reader = csv.reader(file2)
jog_header = next(jog_reader)
walk_header = next(walk_reader)

# Accumulate data from the rows, skip the serial number (first element)
jog_rows = []
for row in jog_reader:
    jog_rows.append([float(i) for i in row[1:]])
walk_rows = []
for row in walk_reader:
    walk_rows.append([float(i) for i in row[1:]])

# Convert to arrays
jog_rows = np.array(jog_rows)
walk_rows = np.array(walk_rows)



# Initialize a numpy array containing 24 (12 x 400) matrices 
# Shape (24, 12, 400)
combined_data = np.zeros([2*M,d,T])

# Take blocks of 400 timesteps from jog_rows and walk_rows to add to the combined data
for i in range(M):
    combined_data[i, :, :] = np.array(jog_rows[400*i:400*(i+1),:]).transpose()
for i in range(M):
    combined_data[(M+i), :, :] = np.array(walk_rows[400*i:400*(i+1),:]).transpose()
    

"""End of Preprocessing"""

# Choose sample trajectories for jogging and walking, to be plotted below
sample_jog_traj = combined_data[0, :, :]
sample_walk_traj = combined_data[2*M-1, :, :]

# Plot and save a figure for the sample jogging trajectory
fig, ax = plt.subplots()

for i in range(d):
    ax.plot(sample_jog_traj[i,:], linewidth = 0.5)
ax.set(xlabel = "Time step", ylabel = "Measurement", title = "Mode 1: jogging")
fig.savefig("jog_fig.png")
plt.close()


# Plot and save a figure for the sample walking trajectory
fig, ax = plt.subplots()

for i in range(d):
    ax.plot(sample_walk_traj[i,:], linewidth = 0.5)
ax.set(xlabel = "Time step", ylabel = "Measurement", title = "Mode 2: walking")
fig.savefig("walk_fig.png")
plt.close()

Vs, Us = subspace_estimation(combined_data,K)
tau = 140
labels, S_original, S = clustering_fast(combined_data, Vs, Us, K, tau, 0)
fig, ax = plt.subplots()
ax.set(xlabel = "Trajectories", ylabel = "Trajectories", title = "Color map of the original matrix")
cax = ax.matshow(S_original, cmap = "gray")
ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
fig.colorbar(cax, ax=ax)
plt.show()
