import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')

# Define your data
baseline_value = 93.77
augmentation_percentages = np.arange(0, 1.2, 0.2)  # From 0 to 1.0 with 0.2 steps
results = [91.35, 93.01, 93.30, 92.66, 90.90, 91.30]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data points
ax.plot(augmentation_percentages, results, marker='o', linestyle='-')

# Add a horizontal line for the baseline value
ax.axhline(y=baseline_value, color='r', linestyle='--', label='Baseline')

# Set labels and title
ax.set_xlabel('Augmentation Percentage', fontsize=14)
ax.set_ylabel('ADD (-S) 0.1', fontsize=14)
ax.set_title('Pix2Pose LMO Augmentation Variation', fontsize=16)

# Add a legend
ax.legend()

# Show the plot
plt.grid(True)
plt.show()
