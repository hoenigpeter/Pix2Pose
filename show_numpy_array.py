import os
import numpy as np
import matplotlib.pyplot as plt

def load_npy_folder(folder_path):
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    file_list.sort()

    images = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        image_array = np.load(file_path)
        images.append(image_array)

    return images

def browse_images(images):
    num_images = len(images)
    for i, image in enumerate(images):
        # Ensure the image_array has dtype as 'uint8' (8-bit unsigned integer) to represent RGB values correctly.
        image_array = image.astype(np.uint8)

        # Display the first image with channels 1 to 3 (R, G, and B)
        rgb_image = image_array[:, :, :3]
        plt.subplot(1, 2, 1)  # Create a subplot for the first image
        plt.imshow(rgb_image)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title('Channels 1 to 3')

        # Display the second image with channels 4 to 6
        channels_4_to_6 = image_array[:, :, 3:6]
        plt.subplot(1, 2, 2)  # Create a subplot for the second image
        plt.imshow(channels_4_to_6)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title('Channels 4 to 6')
        plt.show()

if __name__ == "__main__":
    folder_path = "lmo/train_xyz/01"  # Replace this with the path to your folder containing npy images
    images = load_npy_folder(folder_path)
    if not images:
        print("No npy images found in the folder.")
    else:
        browse_images(images)
