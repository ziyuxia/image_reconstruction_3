import os
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
# from model.encoder_2 import GatingAutoEncoder
from model.resnet_101 import GatingAutoEncoder
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt





def main():
    # Specify the path to the folder containing the images
    # folder_path = "data/minikenetic_train"
    folder_path = "data/minikenetic_test"
    # folder_path = "data/celeba_30000_train"
    # folder_path = "data/celeba_test"
    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to (256, 256)
        transforms.ToTensor(),  # Convert image to tensor

    ])

    # Create an instance of the ImageDataset
    dataset = ImageDataset(folder_path, transform=transform)

    # Create a DataLoader to load the images in batches
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GatingAutoEncoder()

    model = nn.DataParallel(model.cuda())

    state = torch.load('resnet101_weights_minikenetic_1_60_60.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state, strict=False)
    model.eval()


    criterion = torch.nn.MSELoss()


    # Iterate over the data loader to access the images
    losses1 = {}
    losses2 = {}

    for name, batch in dataloader:
        # Perform operations on the batch of images
        img = batch
        img = img.to(device="cuda")
        out1, out2 = model(img)
        loss1 = criterion(out1, img)
        loss2 = criterion(out2, img)
        losses1[name] = loss1.item()
        losses2[name] = loss2.item()










            # example_image = img[0].to(device="cpu")  # Get the input image and its index
            # example_image = example_image.permute(1, 2, 0)  # Reshape tensor for plotting (C, H, W) to (H, W, C)
            #
            # example_image1 = out[0].to(device="cpu")  # Get the ouput image and its index
            # example_image1 = example_image1.permute(1, 2, 0)


            # example_image2 = out2[0].to(device="cpu")  # Get the first image and its index
            # example_image2 = example_image2.permute(1, 2, 0)

            # Plot the image using Matplotlib
            # plt.subplot(1, 2, 1)
            # plt.imshow(example_image.detach().numpy())
            # plt.axis('off')
            #
            # plt.subplot(1, 2, 2)
            # plt.imshow(example_image1.detach().numpy())
            # plt.axis('off')
            # plt.show()

            # plt.imshow(example_image2.detach().numpy())
            # plt.axis('off')
            # plt.show()

    losses1 = sorted(losses1.items(), key=lambda x:x[1])
    losses2 = sorted(losses2.items(), key=lambda x:x[1])
    first_20_items_val = [[x[1] for x in losses1[:20]], [x[1] for x in losses2[:20]]]
    last_20_items_val = [[x[1] for x in losses1[-20:]], [x[1] for x in losses2[-20:]]]
    br1 = np.arange(len(first_20_items_val[0]))
    br2 = [x + 0.4 for x in br1]
    plt.bar(br1, first_20_items_val[0], 0.4, label='gate0')
    plt.bar(br2, first_20_items_val[1], 0.4, label='gate1')
    plt.show()
    plt.bar(br1, last_20_items_val[0], 0.4, label='gate0')
    plt.bar(br2, last_20_items_val[1], 0.4, label='gate1')
    plt.show()




if __name__ == "__main__":
    main()
