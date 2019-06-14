from Functions import *
from Dataset_path import *
# import functions from Functions


# initialize
parameters = init_parameters()
current_epoch = 0
train_loss_list = []
valid_loss_list = []
train_accu_list = []
valid_accu_list = []
print(valid_accuracy(parameters))
learn_rate = 10**-0.62
epoch_num = 5
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
