import os
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
from model.encoder_2 import GatingAutoEncoder
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt





def main():
    # Specify the path to the folder containing the images
    folder_path = "data/test"

    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to (256, 256)
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.4255, 0.3997, 0.3710), # Normalize the tensor
                             (0.2819, 0.2770, 0.2801))
    ])

    # denormalize the image
    unorm = transforms.Normalize(mean=[-0.4255 / 0.2819, -0.3997 / 0.2770, -0.3710 / 0.2801],
                                 std=[1 / 0.2819, 1 / 0.2770, 1 / 0.2801])

    # Create an instance of the ImageDataset
    dataset = ImageDataset(folder_path, transform=transform)

    # Create a DataLoader to load the images in batches
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GatingAutoEncoder()

    model = nn.DataParallel(model.cuda())

    state = torch.load('model_weights_12_400.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    model.train()

    criterion = torch.nn.MSELoss()
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    #calculate the mean and the standard deviation
    imgs = torch.stack([img_t for img_t in dataset], dim=3)
    a=imgs.view(3, -1)
    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)

    # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        count = 0
        total_loss = 0

        for batch in dataloader:
            # Perform operations on the batch of images
            count += 1
            img = batch
            img = img.to(device="cuda")
            out = model(img)


            loss = criterion(out, img)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            real_img = unorm(img)
            real_out = unorm(out)
            real_loss = criterion(real_img, real_out)
            print(real_loss)






            example_image = real_img[0].to(device="cpu")  # Get the input image and its index
            example_image = example_image.permute(1, 2, 0)  # Reshape tensor for plotting (C, H, W) to (H, W, C)
            
            example_image1 = real_out[0].to(device="cpu")  # Get the ouput image and its index
            example_image1 = example_image1.permute(1, 2, 0)


            plt.imshow(example_image.detach().numpy())
            plt.axis('off')
            plt.show()
            
            plt.imshow(example_image1.detach().numpy())
            plt.axis('off')
            plt.show()



        print("total_loss:", total_loss/count)


    torch.save(model.state_dict(), 'model_weights_12_500.pth')
if __name__ == "__main__":
    main()
