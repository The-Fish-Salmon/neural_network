import math
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy
# numpy and math is for functions and algebra
# struct is for unpacking data sets
# matplotlib is for plotting the pictures
# pathlib is for path def


def bypass(x):
    return x


def tanh(x):
    return np.tanh(x)
# defining tan function: change around +-1 but flat above +-2


def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()
# define softmax function: get index over sum of index:input a list and get a list of number that have a sum of 1
# x-x.max is to prevent index getting to big
# the sum have to have () because the exp.sum is a algebra object


def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm)-np.outer(sm, sm)
# diag: give a matrix with only diag have value
# outer: each row times the value of the second string in order


def d_tanh(data):
    return 1/(np.cosh(data))**2


def d_bypass(x):
    return 1


differential = {softmax: d_softmax, tanh: d_tanh, bypass: d_bypass}
d_type = {bypass: 'times', softmax: 'dot', tanh: 'times'}

dimensions = [28 * 28, 100, 10]
activation = [bypass, tanh, softmax]
distribution = [
    {},  # leave it empty!!
    {'b': [0, 0], 'w': [-math.sqrt(6 / (dimensions[0] + dimensions[1])), math.sqrt(6 / (dimensions[0] + dimensions[1]))]},
    {'b': [0, 0], 'w': [-math.sqrt(6 / (dimensions[1] + dimensions[2])), math.sqrt(6 / (dimensions[1] + dimensions[2]))]},
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
    return np.random.rand(dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
# return as a range because the dimension is a range
# give the number of "layer" amount of random number: e.g. 1 st layer will have 28x28
# then use the * (dist[1] - dist[0] + dist[0]) to make it into the dimension we want
# e.g. layer one: 784 zeros, layer two 10 zeros
# initializing b, 2 layers.


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer - 1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
# same as the initializing of b
# but we want a 2 dimensional number so dimensions[layer - 1], dimensions[layer]
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
    l_in = img
    l_out = activation[0](l_in)
    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out, parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
    return l_out


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


h = 0.0001
func = softmax
input_len = 4
for i in range(input_len):
    test_input = np.random.rand(input_len)
    derivative = differential[func](test_input)
    value1 = func(test_input)
    test_input[i] += h
    value2 = func(test_input)
    # print((value2-value1)/h)
    # print(derivative[i])
    # print(derivative[i] - (value2 - value1) / h)

h = 0.0001
func = tanh
input_len = 4
for i in range(input_len):
    test_input = np.random.rand(input_len)
    derivative = differential[func](test_input)
    value1 = func(test_input)
    test_input[i] += h
    value2 = func(test_input)
    # print((value2-value1)/h)
    # print(derivative[i])
    # print(derivative[i] - (value2 - value1) / h)


onehot = np.identity(dimensions[-1])


def sqr_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y-y_pred
    return np.dot(diff, diff)


def grad_parameters(img, lab, parameters):
    l_in_list = [img]
    l_out_list = [activation[0](l_in_list[0])]
    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out_list[layer-1], parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
        l_in_list.append(l_in)
        l_out_list.append(l_out)

    d_layer = -2*(onehot[lab]-l_out_list[-1])

    grad_result = [None]*len(dimensions)
    for layer in range(len(dimensions)-1, 0, -1):
        if d_type[activation[layer]] == 'times':
            d_layer = differential[activation[layer]](l_in_list[layer])*d_layer
        if d_type[activation[layer]] == 'dot':
            d_layer = np.dot(differential[activation[layer]](l_in_list[layer]), d_layer)
        grad_result[layer] = {}
        grad_result[layer]['b'] = d_layer
        grad_result[layer]['w'] = np.outer(l_out_list[layer-1], d_layer)
        d_layer = np.dot(parameters[layer]['w'], d_layer)
    return grad_result


parameters = init_parameters()

h = 0.00001
layer = 2
pname = 'b'
grad_list = []
for i in range(len(parameters[layer][pname])):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
    value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
    test_parameters[layer][pname][i] += h
    value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
    grad_list.append(derivative[i] - (value2 - value1) / h)
# print(np.abs(grad_list).max())

h = 0.00001
layer = 2
pname = 'w'
grad_list = []
for i in range(len(parameters[layer][pname])):
    for j in range(len(parameters[layer][pname][0])):
        img_i = np.random.randint(train_num)
        test_parameters = init_parameters()
        derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
        value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
        test_parameters[layer][pname][i][j] += h
        value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
        grad_list.append(derivative[i][j] - (value2 - value1) / h)
# print(np.abs(grad_list).max())

h = 0.00001
layer = 1
pname = 'b'
grad_list = []
for i in range(len(parameters[layer][pname])):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
    value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
    test_parameters[layer][pname][i] += h
    value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
    grad_list.append(derivative[i] - (value2 - value1) / h)
# print(np.abs(grad_list).max())

h = 0.00001
layer = 1
pname = 'w'
grad_list = []
for i in range(len(parameters[layer][pname])):
    for j in range(len(parameters[layer][pname][0])):
        img_i = np.random.randint(train_num)
        test_parameters = init_parameters()
        derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
        value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
        test_parameters[layer][pname][i][j] += h
        value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
        grad_list.append(derivative[i][j] - (value2 - value1) / h)
# print(np.abs(grad_list).max())


def valid_loss(parameters):
    loss_accu = 0
    for img_i in range(valid_num):
        loss_accu += sqr_loss(valid_img[img_i], valid_lab[img_i], parameters)
    return loss_accu/(valid_num//10000)


def valid_accuracy(parameters):
    correct = [predict(valid_img[img_i], parameters).argmax() == valid_lab[img_i] for img_i in range(valid_num)]
    return correct.count(True)/len(correct)


def train_loss(parameters):
    loss_accu = 0
    for img_i in range(train_num):
        loss_accu += sqr_loss(train_img[img_i], train_lab[img_i], parameters)
    return loss_accu/(train_num/10000)


def train_accuracy(parameters):
    correct = [predict(train_img[img_i], parameters).argmax() == train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True)/len(correct)


def test_accuracy(parameters):
    correct = [predict(test_img[img_i], parameters).argmax() == test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True)/len(correct)


def grad_add(grad1, grad2):
    for layer in range(1, len(grad1)):
        for pname in grad1[layer].keys():
            grad1[layer][pname] += grad2[layer][pname]
    return grad1


def grad_divide(grad, denominator):
    for layer in range(1, len(grad)):
        for pname in grad[layer].keys():
            grad[layer][pname] /= denominator
    return grad


def combine_parameters(parameters, grad, learn_rate):
    parameter_tmp = copy.deepcopy(parameters)
    for layer in range(1, len(parameter_tmp)):
        for pname in parameter_tmp[layer].keys():
            parameter_tmp[layer][pname] -= learn_rate * grad[layer][pname]
    return parameter_tmp


batch_size = 100


def train_batch(current_batch, parameters):
    grad_accu = grad_parameters(train_img[current_batch*batch_size+0], train_lab[current_batch*batch_size+0], parameters)
    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(train_img[current_batch * batch_size + img_i], train_lab[current_batch * batch_size + img_i], parameters)
        grad_add(grad_accu, grad_tmp)
        grad_divide(grad_accu, batch_size)
    return grad_accu


parameters = init_parameters()
current_epoch = 0
train_loss_list = []
valid_loss_list = []
train_accu_list = []
valid_accu_list = []
print(valid_accuracy(parameters))
learn_rate = 10**-0.6
epoch_num = 10
for epoch in range(epoch_num):
    for i in range(train_num//batch_size):
        if i % 100 == 99:
            print('running batch {}/{}'.format(i+1, train_num//batch_size))
        grad_tmp = train_batch(i, parameters)
        parameters = combine_parameters(parameters, grad_tmp, learn_rate)
    current_epoch += 1
    train_loss_list.append(train_loss(parameters))
    train_accu_list.append(train_accuracy(parameters))
    valid_loss_list.append(valid_loss(parameters))
    valid_accu_list.append(valid_accuracy(parameters))

lower = -20
plt.plot(valid_loss_list[lower:], color='black', label='validation loss')
plt.plot(train_loss_list[lower:], color='red', label='train loss')
plt.show()
plt.plot(valid_accu_list[lower:], color='black', label='validation accuracy')
plt.plot(train_accu_list[lower:], color='red', label='train accuracy')
plt.show()

rand_batch = np.random.randint(train_num//batch_size)
grad_lr = train_batch(rand_batch, parameters)

lr_list = []
lower = -2.75
upper = -0.6
step = 0.1
for lr_pow in np.linspace(lower, upper, num = (upper-lower)//step+1):
    learn_rate = 10**lr_pow
    parameters_tmp = combine_parameters(parameters, grad_lr, learn_rate)
    train_loss_tmp = train_loss(parameters_tmp)
    lr_list.append([lr_pow, train_loss_tmp])

upper = len(lr_list)
plt.plot(np.array(lr_list)[:upper, 0], np.array(lr_list)[:upper, 1], color='black')
plt.show()