
The directory 'cifar10_experiments' contains the code necessary for running the experiments. populate_dirs.py is used to generate images with the GAN, and create the various sized datasets. exp.py is used to sort generated images and form the similarity matrix shown in figure 4 of the Final Project paper. populate_dirs_training_comp.py and exp_training_comp.py are used to run the experiments necessary for comparing generated images to their nearest neighbor in the training set as shown in Figure 5 of the Final Project Paper.

The other .py files in the directory come from a pytorch project I cited in the paper and the progressive GAN code. They hold the VGG network and some utility code for loading images and such

Email awreed@asu.edu for the .pkl files containing pretrained networks. They are too big to upload to github in any reasonable amount of time
