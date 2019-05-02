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
number_unique_images = 2 
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
   
 
    image_grid = np.zeros((height*number_of_samples, width*sqrt_support_size, channels), dtype=np.uint8)
    print(image_grid.shape)
   
    for i in range(0, number_of_samples):
        gen_files = glob.glob(GEN_DIR + 'sample' + str(i) + '/' + '*.png')
        j = 0
        for f in gen_files:
            img = utils.load_image(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_grid[i*height:(i+1)*height, j*width:(j+1)*width, :] = img[:, :, :] 
            j = j + 1
        
    PIL.Image.fromarray(image_grid, 'RGB').save(OUT_DIR + 'image_grid.png') 

if __name__ == '__main__':
    find_nearest_neighbor()
        
