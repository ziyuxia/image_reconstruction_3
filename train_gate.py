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
    folder_path = "data/minikenetic_test_50"
    # folder_path = "data/celeba_30000_train"
    # folder_path = "data/celeba_test"
    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to (256, 256)
        transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize((0.4255, 0.3997, 0.3710),
        #                      (0.2819, 0.2770, 0.2801))
    ])

    # unorm = transforms.Normalize(mean=[-0.4255 / 0.2819, -0.3997 / 0.2770, -0.3710 / 0.2801],
    #                              std=[1 / 0.2819, 1 / 0.2770, 1 / 0.2801])

    # Create an instance of the ImageDataset
    dataset = ImageDataset(folder_path, transform=transform)

    # Create a DataLoader to load the images in batches
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GatingAutoEncoder()

    model = nn.DataParallel(model.cuda())

    state = torch.load('resnet101_weights_minikenetic_1_60_60.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state, strict=False)
    model.train()
    for param in model.module.encoder1.parameters():
        param.requires_grad = False
    model.module.encoder1.eval()
    for param in model.module.decoder1.parameters():
        param.requires_grad = False
    model.module.decoder1.eval()




    criterion = torch.nn.MSELoss()
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



    # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        count = 0
        total_loss = 0

        for _, batch in dataloader:
            # Perform operations on the batch of images
            count += 1
            img = batch
            img = img.to(device="cuda")
            out, gate = model(img)



            loss = criterion(out, img)
            # loss = loss1 + loss2
            # total_loss1 += loss1
            # total_loss2 += loss2
            total_loss += loss
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # real_img = unorm(img)
            # real_out = unorm(out)
            # real_loss = criterion(real_img, real_out)
            # real_total_loss += real_loss
            # print(real_loss)






            # example_image = img[0].to(device="cpu")  # Get the input image and its index
            # example_image = example_image.permute(1, 2, 0)  # Reshape tensor for plotting (C, H, W) to (H, W, C)
            #
            # example_image1 = out[0].to(device="cpu")  # Get the ouput image and its index
            # example_image1 = example_image1.permute(1, 2, 0)

            # example_image1 = out1[0].to(device="cpu")  # Get the first image and its index
            # example_image1 = example_image1.permute(1, 2, 0)  # Reshape tensor for plotting (C, H, W) to (H, W, C)
            #
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

            # plt.imshow(example_image1.detach().numpy())
            # plt.axis('off')
            # plt.show()
            #
            # plt.imshow(example_image2.detach().numpy())
            # plt.axis('off')
            # plt.show()
        # print("total_loss1:", total_loss1/count)
        # print("total_loss2: {:.8f}".format(total_loss2/count))
        print("total_loss: {:.8f}".format(total_loss / count))

        # print("real_ total_loss:", real_total_loss/count)



    torch.save(model.state_dict(), 'resnet101_weights_minikenetic_1_70_60.pth')
if __name__ == "__main__":
    main()
