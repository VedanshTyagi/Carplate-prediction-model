import matplotlib
import pickle as pkl
import numpy as np
import math 
import csv

def relu(x):
    return np.maximum(0, x)

features = np.array(pkl.load(open('features.pkl', 'rb')))
features1 = np.array(pkl.load(open('features1.pkl', 'rb')))
keys = pkl.load(open('keys.pkl', 'rb'))
keys = np.array([int(i) for i in keys])

neurons1 = 256
neurons2 = 128
neurons3 = 64
weights1  = np.random.randn(neurons1, 420)
bias1 = np.random.randn(neurons1, 1) 
weights2 = np.random.randn(neurons2, neurons1)
bias2 = np.random.randn(neurons2, 1)
weights3 = np.random.randn(neurons3, neurons2)
bias3 = np.random.randn(neurons3, 1)
weights4 = np.random.randn(1, neurons3)
bias4 = np.random.randn(1, 1)


epochs = math.floor(51635/400)
b = 400 #batch size
lr = 10 #learning rate
for n in range(epochs):
    cost = 0
    if n>100:lr=1
    for feature,price in zip(features[b*n:b*(n+1)],keys[b*n:b*(n+1)]):

        # front propogation
        layer1 = relu((weights1 @ feature.reshape(420,1)) + bias1)
        layer2 = relu((weights2 @ layer1) + bias2)
        layer3 = relu((weights3 @ layer2) + bias3)
        pre = ((weights4 @ layer3) + bias4)[0][0]
        cost += (2*abs(price - pre))/(abs(price) + abs(pre))

        #back propogation
        dprice = 2*price * np.sign(pre - price)/ (pre + price) ** 2
        dweights4 = layer3.T * dprice
        dbias4 = dprice
        dlayer3 = (weights4.T * dprice) * (layer3>0)
        dweights3 = dlayer3 @ layer2.T
        dbias3 = dlayer3
        dlayer2 = (weights3.T @ dlayer3) * (layer2>0)
        dweights2 = dlayer2 @ layer1.T
        dbias2 = dlayer2
        dlayer1 = (weights2.T @ dlayer2) * (layer1 > 0)
        dweights1 = dlayer1 @ feature.reshape(1, 420)
        dbias1 = dlayer1 

        # update weights and biases
        weights1 -= lr * dweights1
        bias1 -= lr * dbias1
        weights2 -= lr * dweights2
        bias2 -= lr * dbias2
        weights3 -= lr * dweights3
        bias3 -= lr * dbias3
        weights4 -= lr * dweights4
        bias4 -= lr * dbias4
    else:
        print(f"Epoch {n}, Cost:", cost*100/b)


price=[]
# Predicting prices for the test set
for feature in features1:
    layer1 = relu((weights1 @ feature.reshape(420,1)) + bias1)
    layer2 = relu((weights2 @ layer1) + bias2)
    layer3 = relu((weights3 @ layer2) + bias3)
    pre = ((weights4 @ layer3) + bias4)
    price.append(pre[0][0])

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'price'])
    for i, p in zip(range(51636,59331), price):
        writer.writerow([i, p])