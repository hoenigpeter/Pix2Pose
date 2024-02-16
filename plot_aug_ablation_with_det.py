import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')

# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
augmentation_percentages = np.arange(0, 1.2, 0.2)  # From 0 to 1.0 with 0.2 steps

# Define your data for Pix2Pose
baseline_value_pix2pose = 59.24
results_pix2pose = [60.31, 62.95, 63.96, 61.78, 61.19, 58.04]
y_limits_pix2pose = (0, 80)  # Set y-axis limits for Pix2Pose

# Plot the data points for Pix2Pose
ax[0].plot(augmentation_percentages, results_pix2pose, color='blue', marker='o', linestyle='-')
ax[0].axhline(y=baseline_value_pix2pose, color='purple', linestyle='--', label='Baseline')
ax[0].set_xlabel('Augmentation Probability Scaling $\lambda$', fontsize=12)
ax[0].set_ylabel('ADD(-S)', fontsize=12)
ax[0].set_title('Pix2Pose', fontsize=14)
ax[0].legend(fontsize=12)
ax[0].set_ylim(y_limits_pix2pose)  # Set y-axis limits for Pix2Pose

# Define your data for Faster R-CNN
baseline_value_faster_rcnn = 67.75
results_faster_rcnn = [55.74, 63.73, 64.21, 63.10, 62.64, 61.24]
y_limits_faster_rcnn = (0, 80)  # Set y-axis limits for Faster R-CNN

# Plot the data points for Faster R-CNN
ax[1].plot(augmentation_percentages, results_faster_rcnn, color='red', marker='o', linestyle='-')
ax[1].axhline(y=baseline_value_faster_rcnn, color='green', linestyle='--', label='Baseline')
ax[1].set_xlabel('Augmentation Probability Scaling $\lambda$', fontsize=12)
ax[1].set_ylabel('mAP', fontsize=12)
ax[1].set_title('Faster R-CNN', fontsize=14)
ax[1].legend(fontsize=12)
ax[1].set_ylim(y_limits_faster_rcnn)  # Set y-axis limits for Faster R-CNN

# Show the plot
plt.grid(False)
plt.savefig('augmentation_probability.png', dpi=300, bbox_inches='tight')
