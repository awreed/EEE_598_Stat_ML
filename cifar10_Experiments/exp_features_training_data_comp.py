import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np
import time
import cv2
import PIL.Image
import math

import vgg
import utils
import glob

GEN_DIR = './generated_images/'
TRAIN_DIR = './training_images/'
OUT_DIR = './output_images/'
SEED = 35
number_of_samples = 20
height = 32
width = 32
channels = 3
number_unique_images = [2, 8, 32, 128, 256, 512, 1024]  

for number in number_unique_images:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    VGG = vgg.VGG16().to(device)

    MSELoss = nn.MSELoss().to(device)
    
    imagenet_neg_mean = torch.tensor([-103.939, -119.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).to(device)#subtract the mean of imagenet images
    image_grid = np.zeros((height*number_of_samples, width * 2, channels), dtype=np.uint8)
    gen_files = glob.glob(GEN_DIR + 'number' + str(number) + '/*.png')

    train_files = glob.glob(TRAIN_DIR + 'number' + str(number) + '/*.png')
    row = 0
    average_dist = 0
    for gf in gen_files:#for each generated image (20 total)
        gen_img = utils.load_image(gf)
        gen_img_tensor = utils.itot(gen_img).to(device)
        gen_img_tensor = gen_img_tensor.add(imagenet_neg_mean)
        gen_features = VGG(gen_img_tensor)
        min_dist = 0 
        first = True
        nn_index = 0
        count = 0
        for tf in train_files:#find the closest image in the training set
            t_img = utils.load_image(tf)
            t_img_tensor = utils.itot(t_img).to(device)
            t_img_tensor = t_img_tensor.add(imagenet_neg_mean)
            t_features = VGG(t_img_tensor)
            dist = MSELoss(gen_features['relu1_2'], t_features['relu1_2'])
            if first:
                min_dist = dist.item()
                first = False
            if(dist.item() < min_dist):
                nn_index = count 
                min_dist = dist.item()
            count += 1
        img1 = utils.load_image(gf)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = utils.load_image(train_files[nn_index])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)    
        
        image_grid[row*height:(row+1)*height, 0:width, :] = img1
        image_grid[row*height:(row+1)*height, width:width*2, :] = img2
        row += 1
        average_dist += min_dist
    average_dist = average_dist/number_of_samples 
    print(average_dist)
    PIL.Image.fromarray(image_grid, 'RGB').save(OUT_DIR + 'number' + str(number) + '/' + 'image_grid.png')
         
 
   
       
