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
number_unique_images = 256 
sqrt_support_size = math.ceil(math.sqrt(-1 * math.log(0.5) * 2 * number_unique_images)) 

def find_nearest_neighbor():
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
   
 
   
    for i in range(0, number_of_samples):#For each sample (20 total)
        gen_files = glob.glob(GEN_DIR + 'sample' + str(i) + '/' + '*.png')
        image_grid = np.zeros((height*sqrt_support_size, width *sqrt_support_size, channels), dtype=np.uint8)
        row = 0
        for row_leader in gen_files:#Each unique image gets its own row on the image grid
            row_list = []
            row_leader_img = utils.load_image(row_leader)
            row_leader_tensor = utils.itot(row_leader_img).to(device)
            row_leader_tensor = row_leader_tensor.add(imagenet_neg_mean)
            row_leader_features = VGG(row_leader_tensor)
            for f in gen_files:#compare the "row leader" with every other image in the sample
                 
                img = utils.load_image(f)
                img_tensor = utils.itot(img).to(device)
                img_tensor = img_tensor.add(imagenet_neg_mean)
                img_features = VGG(img_tensor)
                
                row_list.append((MSELoss(row_leader_features['relu1_2'], img_features['relu1_2']).item(), f))#use VGG featuers to find nearest neighbor between row leader and every other image
                
            row_list.sort()#sort the list by distance
            for col in range(0, len(row_list)):#place them on the image grid in order
                img = utils.load_image(row_list[col][1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_grid[row*height:(row+1)*height, col*width:(col+1)*width, :] = img[:, :, :] 
            row_list.clear()
            row = row + 1
        PIL.Image.fromarray(image_grid, 'RGB').save(OUT_DIR + 'sample' + str(i) +  'image_grid.png') 

if __name__ == '__main__':
    find_nearest_neighbor()
        
