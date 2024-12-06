import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Create a simple plot
plt.figure(figsize=(4, 4))
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")
plt.savefig("output_plot.png")  # Save the plot to a file

# Load the uploaded CSV file
file_path = 'merged_data.csv'
data = pd.read_csv(file_path)

# Extract the 'Signed Volume' column to use as trade sizes (dQ_t)
trade_sizes = data['Signed Volume'].values

# Parameters
time_steps = len(trade_sizes)  # Use the number of rows in the dataset as time steps
beta = 0.1                     # Exponential decay rate
lambda_param = 0.01            # Price impact parameter
p = 0.5                        # Nonlinear power for AFS model

# Linear OW Model: Compute J_t
J_t_linear = np.zeros(time_steps)
for t in range(1, time_steps):
    J_t_linear[t] = np.exp(-beta) * J_t_linear[t - 1] + trade_sizes[t]

# Nonlinear AFS Model: Compute I_t
J_t_nonlinear = np.zeros(time_steps)
I_t_nonlinear = np.zeros(time_steps)
for t in range(1, time_steps):
    J_t_nonlinear[t] = np.exp(-beta) * J_t_nonlinear[t - 1] + trade_sizes[t]
    I_t_nonlinear[t] = lambda_param * abs(J_t_nonlinear[t]) ** p

# Plotting
plt.figure(figsize=(12, 6))

# Linear OW Model: Distribution of price impact
plt.subplot(1, 2, 1)
plt.hist(J_t_linear, bins=30, alpha=0.7, color='blue', label='Linear OW Model')
plt.title("Price Impact Distribution (Linear OW Model)")
plt.xlabel("Price Impact (J_t)")
plt.ylabel("Frequency")
plt.legend()

# Nonlinear AFS Model: Distribution of price impact
plt.subplot(1, 2, 2)
plt.hist(I_t_nonlinear, bins=30, alpha=0.7, color='orange', label='Nonlinear AFS Model')
plt.title("Price Impact Distribution (Nonlinear AFS Model)")
plt.xlabel("Price Impact (I_t)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig("output_price_impact.png")  # Save the figure to a file, no need to call plt.show()
