# finding the best leaning rate
from neural_network import *
import matplotlib.pyplot as plt
import copy

lr_list = []
lower = -2.75
upper = -0.6
step = 0.1
for lr_pow in np.linspace(lower, upper, num=int((upper-lower)//step+1)):
    learn_rate = 10**lr_pow
    parameters_tmp = combine_parameters(parameters, grad_lr, learn_rate)
    train_loss_tmp = train_loss(parameters_tmp)
    lr_list.append([lr_pow, train_loss_tmp])

upper = len(lr_list)
plt.plot(np.array(lr_list)[:upper, 0], np.array(lr_list)[:upper, 1], color='black')
plt.show()
