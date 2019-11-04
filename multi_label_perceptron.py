# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from readFile import get3ClassData
import numpy as np

# global
epochs = 500
lr = 0.001
print("Example of multi-label perceptron ")

def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels.astype(int)]
  return one_hot


def compute_loss(predict_total, y):
    y_index = y.argmax(axis=1)
    predict_index = predict_total.argmax(axis=1)
    right_count = (y_index==predict_index).sum()
    acc = (1.0*right_count) / y.shape[0]
    temp1 = np.array([predict_total[i,j] for i, j in enumerate(predict_index)])
    temp2 = np.array([predict_total[i,j] for i,j in enumerate(y_index)])
    loss = (temp1 - temp2).sum()
    return loss, acc




total_loss = []
total_acc = []
np.random.seed(0)
x, y = get3ClassData()
y = transform_one_hot(y.astype(int))
x_dim, y_dim = x.shape[1], y.shape[1]
w = np.random.standard_normal([x_dim, y_dim])  # mean=0, stdev=1
b = np.ones(y_dim)  # init b
predict_total = x.dot(w) + b

# loss before training
loss, acc = compute_loss(predict_total, y)
print("before training, loss = %f, acc = %f" % (loss, acc))
total_loss.append(loss)

# 进入epoch
for epoch in range(epochs):
    shuffle_index = np.random.permutation(x.shape[0])
    x_shuffle = x[shuffle_index, :]
    y_shuffle = y[shuffle_index, :]
    # update param for each x_k
    for k, x_k in enumerate(x_shuffle):
        predict_k = np.argmax(x_k.dot(w) + b)
        y_k = np.argmax(y_shuffle[k])
        
        if predict_k != y_k:  # predict wrong, and update param
            w[:, predict_k] = w[:, predict_k] - lr * x_k
            b[predict_k] = b[predict_k] - lr * 1
            w[:,y_k] = w[:,y_k] + lr * x_k 
            b[y_k] = b[y_k] + lr * 1

        else:  # predict right
            pass
    # loss of the epoch
    predict_total = x_shuffle.dot(w) + b
    loss, acc = compute_loss(predict_total, y_shuffle)
    total_loss.append(loss)
    total_acc.append(acc)
    if (epoch+1) % 10 == 0:
        print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))

        
        

# /test
#w[:,2]
#right_count = (y_index==predict_index.sum())
# temp1, temp2 = np.random.shuffle()
# /test
