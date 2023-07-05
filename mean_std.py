# Code to calculate the mean and the std for the test image dataset

import os
import numpy as np
from PIL import Image
# Set the path to the folder containing the RGB images
folder_path = "data/test"

# Create an empty list to store the resized images
resized_images = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        
        # Resize the image to 256x256
        resized_image = image.resize((256, 256))
        
        # Convert the resized image to a numpy array
        resized_array = np.array(resized_image)
        
        # Add the resized image array to the list
        resized_images.append(resized_array)

# Convert the list of resized images to a single numpy array
image_array = np.stack(resized_images)/255


print(image_array.shape)

mean = np.mean(image_array, axis = (0, 1, 2))
std = np.std(image_array, axis = (0, 1, 2))
count = 0
print(mean)
print(std)
# sum1 = 0
# sum2 = 0
# for i in range(image_array.shape[0]):
#     for j in range(image_array.shape[1]):
#         for k in range(image_array.shape[2]):
#             sum1+=image_array[i][j][k][0]
#             sum2+=(image_array[i][j][k][0]-mean[0])**2
#             count+=1
# print((sum1/count)**0.5)
# print(sum2/count)



# Convert the mean and average tensors to numpy arrays
# mean_array = mean_tensor.numpy()
# average_array = average_tensor.numpy()

# # Print the mean and average values
# print("Mean:", mean_array)
# print("Average:", average_array)
