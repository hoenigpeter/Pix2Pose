import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')

# Define your data
baseline_value = 70.15
augmentation_percentages = np.arange(0, 1.2, 0.2)  # From 0 to 1.0 with 0.2 steps
results = [68.27, 75.98, 74.47, 72.65, 74.23, 67.83]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data points
ax.plot(augmentation_percentages, results, marker='o', linestyle='-')

# Add a horizontal line for the baseline value
ax.axhline(y=baseline_value, color='r', linestyle='--', label='Baseline')

# Set labels and title
ax.set_xlabel('Augmentation Probability Scaling $\lambda$', fontsize=12)
ax.set_ylabel('ADD(-S)', fontsize=12)
#ax.set_title('a) TLESS', fontsize=14)

# Add a legend
ax.legend(fontsize=12)

# Show the plot
plt.grid(False)
fig.savefig('tless_augmentation.png', dpi=300, bbox_inches='tight')

# Close the figure to free up resources (optional)
plt.close(fig)