import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')

# Create a figure and axis
fig, ax = plt.subplots(1,2, figsize=(8,3))
augmentation_percentages = np.arange(0, 1.2, 0.2)  # From 0 to 1.0 with 0.2 steps

# Define your data
baseline_value = 70.15
results = [68.27, 75.98, 74.47, 72.65, 74.23, 67.83]

# Plot the data points
ax[0].plot(augmentation_percentages, results, color='blue', marker='o', linestyle='-')

# Add a horizontal line for the baseline value
ax[0].axhline(y=baseline_value, color='purple', linestyle='--', label='Baseline')

# Set labels and title
ax[0].set_xlabel('Augmentation Probability Scaling $\lambda$', fontsize=12)
ax[0].set_ylabel('ADD(-S)', fontsize=12)
ax[0].set_title('Pix2Pose', fontsize=14)

# Add a legend
ax[0].legend(fontsize=12)

# 55,74
# 63,73
# 64,21
# 63,10
# 62,64
# 61,24

baseline_value = 67.75
results = [55.74, 63.73, 64.21, 63.10, 62.64, 61.24]

# Plot the data points
ax[1].plot(augmentation_percentages, results, color='red', marker='o', linestyle='-')

# Add a horizontal line for the baseline value
ax[1].axhline(y=baseline_value, color='green', linestyle='--', label='Baseline')

# Set labels and title
ax[1].set_xlabel('Augmentation Probability Scaling $\lambda$', fontsize=12)
ax[1].set_ylabel('mAP', fontsize=12)
ax[1].set_title('Faster R-CNN', fontsize=14)

# Add a legend
ax[1].legend(fontsize=12)

# Show the plot
plt.grid(False)
plt.savefig('your_figure.png', dpi=300, bbox_inches='tight')  # Adjust filename and DPI as needed
