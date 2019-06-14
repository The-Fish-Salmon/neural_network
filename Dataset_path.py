# Route to datasets
from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt
import copy
# struct is for unpacking data sets
# matplotlib is for plotting the pictures
# pathlib is for path def

dataset_path = Path('datasets/MNIST/')
train_img_path = dataset_path/'train-images.idx3-ubyte'
train_lab_path = dataset_path/'train-labels.idx1-ubyte'
test_img_path = dataset_path/'t10k-images.idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels.idx1-ubyte'
# def path for training img and label, testing img and lab

train_num = 50000
valid_num = 10000
test_num = 10000
# def number of training img and test img, separate 10000 img from training for validation
# training: give picture and label for models to learn
# validation: give only you the label: function as a quiz --> picking of the best model during training
# test: label is hidden untouchable to both you and model
# importance of test: prevent model from "reciting" the training and eliminate randomness in
# and secondary information leak of label during the "quiz"

with open(train_img_path, 'rb') as f:
    # make f the binary read from training img
    struct.unpack('>4i', f.read(16))
    # unpack the first 4 unused value: stated in the description page of data set
    # unpack f(the binary img) 16 byte
    # >: bit stored direction is greater first;
    # 4: 4 values
    # i: as integer
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28)/255
    # read from f, using data type: no sign(u), integer(int), 1 byte (8)
    # because all img is linked, we need to separate it to 28 x 28 imgs
    # feed back a 2 dimensional matrix where each row is an img
    # using tmp_img to separate 10000 img from training into validation
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]
    # [:train_num] : before train_num; in this case train_img is the all the img before the 50000th img
    # [train_num:] : after train_num; in this case the 10000 img after the 50000 img
with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28)/255
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    # labs only have 2 unused value(8 byte), said in the data set description
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]
with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

'''
# define function that plot a img with its label printed with its given index(counting number)
def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    # draw this 28x28 img in gray scale given it to be the [index]th img.
    print('label : {}'.format(train_lab[index]))
    # print the label in the format of: "label : the label of this index"


def show_valid(index):
    plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
    print('label : {}'.format(valid_lab[index]))


def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label : {}'.format(test_lab[index]))
'''