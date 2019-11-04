# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:55:45 2019
# it works
@author: admin
"""

from readFile import get3ClassData
import numpy as np
import matplotlib.pyplot as plt


colors = ['b','g','r','orange']
# global
epochs = 100
lr = 0.001
print("Example of multi-label perceptron ")

def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels.astype(int)]
  return one_hot

def compute_total_loss(predict_total, y):
    # loss
    loss_true = (np.log(predict_total) * y).sum()
    loss_false= (np.log(1 - predict_total) * y).sum()
    loss = -1 * (loss_true + loss_false)
    # acc
    y_index = y.argmax(axis=1)
    predict_index = predict_total.argmax(axis=1)
    right_count = (y_index==predict_index).sum()
    acc = (1.0*right_count) / y.shape[0]
    
    return loss, acc


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z))) 

def dsigmoid(z):
    return (z>0) * (sigmoid(z) * (1 - sigmoid(z)))


def relu(z):
    return (z>0) * z

def drelu(z):
    return (z>0) * 1

def act(z):
    return sigmoid(z)

def dact(z):
    return dsigmoid(z)



total_loss = []
total_acc = []
dw_list = []
np.random.seed(0)
x, y = get3ClassData()
y_raw = y
y = transform_one_hot(y.astype(int))
x_dim, y_dim = x.shape[1], y.shape[1]
w = np.random.standard_normal([x_dim, y_dim])  # mean=0, stdev=1
b = np.ones(y_dim)   #  init b

# loss before training
predict_total = act(x.dot(w) + b)
loss, acc = compute_total_loss(predict_total, y)

print("before training, loss = %f, acc = %f" % (loss, acc))
total_loss.append(loss)
total_acc.append(acc)

def step():
    # shuffle
    global x,w,b
    shuffle_index = np.random.permutation(x.shape[0])
    x_shuffle = x[shuffle_index, :]
    y_shuffle = y[shuffle_index, :]
    # update param for each x_k
    for k, x_k in enumerate(x_shuffle):
        # transfer 2 matrix
        x_k = x_k.reshape(-1, len(x_k))  # (1,2)
        y_k = y_shuffle[k].reshape(-1, len(y_shuffle[k]))   # (1,3)
        # forward
        pred_k = act(x_k.dot(w) + b)  #(1,3)
        loss_k = -1.0 * ((y_k * np.log(pred_k) + (1 - y_k) * np.log(1 - pred_k))) 
        # backward
        dw = x_k.T.dot(pred_k - y_k)   # (2,3)=(2,1)*(1,3)
        db = (pred_k - y_k).sum(axis=0)             # (3)
        # update
        w -= lr * dw
        b -= lr * db
        dw_list.append(dw)

    # compute loss for the epoch
    predict_total = act(x_shuffle.dot(w) + b)
    loss, acc = compute_total_loss(predict_total, y_shuffle)
    return loss,acc   

def plot_step():

    plt.ion()
    plt.cla()
    plt.subplot(221)
    plt.scatter(range(0,len(total_acc),10),total_acc[::10],color='blue')  # acc plot
    plt.plot(range(len(total_acc)),total_acc,color='blue')
    plt.subplot(222)
    plt.scatter(range(0,len(total_loss),10),total_loss[::10],color='red')  # loss plot
    plt.plot(range(len(total_loss)),total_loss,color='red')
    plt.subplot(223)
    for i in range(y.shape[1]):
        xx = x[y_raw==i]
        plt.scatter(xx[:,0],xx[:,1],s=5)

    # plt.scatter(x[:,0],x[:,1],s=5)
    plt.pause(0.02)
    # plt.ioff()
    plt.show()


for epoch in range(epochs):
   
    loss,acc = step()
    plot_step()
    total_loss.append(loss)
    total_acc.append(acc)
    if (epoch+1) % 10 == 0:
        print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))



