import tensorflow as tf
import misc
import numpy as np
import PIL.Image
import pickle
import os
import math
import shutil

def clear_dir(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

number_unique_images = 256 

pkl_dir = '../results/cifar' + str(number_unique_images) + '/network-snapshot-011756.pkl'
gen_dir = './generated_images/'
train_dir = './training_images/images/'
cifar10_dir = '../../cifar-10-batches-py/'

sqrt_support_size = math.ceil(math.sqrt(-1 * math.log(0.5) * 2 * number_unique_images)) 
sample_size = 20
random_state = 123

clear_dir(gen_dir)#clean up 
clear_dir(train_dir)

sess = tf.Session()

with sess.as_default():#create generated images
    G, D, Gs = misc.load_pkl(pkl_dir)
    for i in range(0, sample_size):
        path = gen_dir + "sample" + str(i) + "/"
        os.mkdir(path)
        for idx in range(0, sqrt_support_size):
            latent = np.random.randn(1, *Gs.input_shapes[0][1:])
            label = np.zeros([latent.shape[0]] + Gs.input_shapes[1][1:])
            image = Gs.run(latent, label)
            image = np.clip(np.rint((image + 1.0) /2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
            image = image.transpose(0, 2, 3, 1)
            print(image[0].shape)
            PIL.Image.fromarray(image[0], 'RGB').save(gen_dir + 'sample' + str(i) + '/' + 'image' + str(idx) + '.png')
       
#create training dataset images
images = []
labels = []

for batch in range(1, 6):
    with open(os.path.join(cifar10_dir, 'data_batch_%d' % batch), 'rb') as file: 
        data = pickle.load(file, encoding='latin1')
    images.append(data['data'].reshape(-1, 3, 32, 32))
    labels.append(data['labels'])
images = np.concatenate(images)
labels = np.concatenate(labels)

assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
assert labels.shape == (50000,) and labels.dtype == np.int64
assert np.min(images) == 0 and np.max(images) == 255
assert np.min(labels) == 0 and np.max(labels) == 9

dataset_size = 16384
number_repeats = int(dataset_size/number_unique_images)
order = np.arange(images.shape[0])
np.random.RandomState(random_state).shuffle(order)
for idx in range(0, number_unique_images):
    img = images[order[idx]]
    img = img.transpose(1, 2, 0)
    print(img.shape)
    PIL.Image.fromarray(img, 'RGB').save(train_dir + 'image' + str(idx) + '.png')


