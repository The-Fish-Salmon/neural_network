import math
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
# numpy and math is for functions and algebra
# struct is for unpacking data sets
# matplotlib is for plotting the pictures
# pathlib is for path def


def tanh(x):
    return np.tanh(x)
# defining tan function: change around +-1 but flat above +-2


def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()
# define softmax function: get index over sum of index:input a list and get a list of number that have a sum of 1
# x-x.max is to prevent index getting to big
# the sum have to have () because the exp.sum is a algebra object


dimensions = [28 * 28, 10]
activation = [tanh, softmax]
distribution = [
    {'b': [0, 0]},
    {'b': [0, 0], 'w': [-math.sqrt(6 / (dimensions[0] + dimensions[1])), math.sqrt(6 / (dimensions[0] + dimensions[1]))]},
]
# l0 = A(data + b0)
# output = A' (l0 * w1 + b1)
# A = tan
# A' = softmax
# b0:784 ; b1: 10; w1:[784,10] because goal is to process 28x28 pixel picture of hand written number
# bo and b1 is a constant that can change depending on what to process
# initial value should be: b0,b1 --> 0  ; w1 square root +- 6/784+10 which is dimension 0 and dimension 1 in the list
# defining range 2 layers one with b and one with b and w


def init_parameters_b(layer):
    dist = distribution[layer]['b']
# find dist as the distribution number "layer" and find b
    return np.random.rand(dimensions[layer]) * (dist[1] - dist[0] + dist[0])
# return as a range because the dimension is a range
# give the number of "layer" amount of random number: e.g. 1 st layer will have 28x28
# then use the * (dist[1] - dist[0] + dist[0]) to make it into the dimension we want
# e.g. layer one: 784 zeros, layer two 10 zeros
# initializing b, 2 layers.


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer - 1], dimensions[layer]) * (dist[1] - dist[0] + dist[0])
# same as the initializing of b
#  But we want a 2 dimensional number so dimensions[layer - 1], dimensions[layer]
# since there is not w in layer 0, no need to worry about layer -1


# initializing parameter as a whole
def init_parameters():
    parameter = []
    # make a list to store the initializing
    for i in range(len(distribution)):
        # make i a number
        layer_parameter = {}
        # make a empty set for layers in parameter
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
        # put the layer into parameter
    return parameter
# result will be: 0th number b having 784 zeros || 1st number b having 10 zeros and w having 784 x 10 random matrix


parameters = init_parameters()
# initialize


def predict(img, parameters):
    l0_in = img + parameters[0]['b']
    # data + b0
    l0_out = activation[0](l0_in)
    # run function tan the 0th layer
    l1_in = np.dot(l0_out, parameters[1]['w']) + parameters[1]['b']
    # get the input of the softmax function the 1st layer
    # np,dot is combining 2 arrays this case: the w and the l0 output
    l1_out = activation[1](l1_in)
    # run through the softmax function
    return l1_out


predict(np.random.rand(784), parameters).argmax()
# input with random number and see result
# argmax is to find the most possible one

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
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28)
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
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28)
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    # labs only have 2 unused value(8 byte), said in the data set description
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]
with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)


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


# show a random img from training, validation, and testing and print its label
show_train(np.random.randint(train_num))
plt.show()
show_valid(np.random.randint(valid_num))
plt.show()
show_test(np.random.randint(test_num))
plt.show()