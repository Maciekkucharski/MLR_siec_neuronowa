import numpy as np
import pandas as pd
from numpy.random import permutation


def init_parameters():
    weight1 = np.random.rand(10, 784) - 0.5
    bias1 = np.random.rand(10, 1) - 0.5
    weight2 = np.random.rand(10, 10) - 0.5
    bias2 = np.random.rand(10, 1) - 0.5
    old_weight_1 = weight1.copy()
    old_bias_1 = bias1.copy()
    old_weight_2 = weight2.copy()
    old_bias_2 = bias2.copy()
    return weight1, bias1, weight2, bias2, old_weight_1, old_bias_1, old_weight_2, old_bias_2


def update_parameters(weight_1, bias_1, weight_2, bias_2, old_weight_1, old_bias_1, old_weight_2, old_bias_2, dW1, db1,
                      dW2, db2, learning_rate, momentum):
    copy_weight_1 = np.copy(weight_1)
    copy_bias_1 = np.copy(bias_1)
    copy_weight_2 = np.copy(weight_2)
    copy_bias_2 = np.copy(bias_2)
    weight_1 = old_weight_1 * momentum + weight_1 - learning_rate * dW1
    bias_1 = bias_1 - learning_rate * db1 + old_bias_1 * momentum
    weight_2 = weight_2 - learning_rate * dW2 + old_weight_2 * momentum
    bias_2 = bias_2 - learning_rate * db2 + old_bias_2 * momentum
    return weight_1, bias_1, weight_2, bias_2, copy_weight_1, copy_bias_1, copy_weight_2, copy_bias_2


def relu(Z):
    return np.maximum(Z, 0)


# bool representative is ether 0 or 1
def relu_deriv(Z):
    return Z > 0


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# is supposed to find the predictions and create appropriate matrix
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def forward_propagation(weight_1, bias1, weight_2, bias_2, X):
    Z1 = weight_1.dot(X) + bias1
    A1 = relu(Z1)
    Z2 = weight_2.dot(A1) + bias_2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_propagation(Z1, activation_in, Z2, activation_out, weight_1, weight_2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = activation_out - one_hot_Y
    dW2 = 1 / m * dZ2.dot(activation_in.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = weight_2.T.dot(dZ2) * relu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# get predicted values(the ones with most probability is going to get round the predicted values)
def get_predictions(activation_out):
    return np.argmax(activation_out, 0)


# dividing right predictions by number of of predictions in order to get accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent_with_momentum(X, Y, epoch, learning_rate=0.1, momentum=0.9):
    weight_1, bias_1, weight_2, bias_2, old_weight_1, old_bias_1, old_weight_2, old_bias_2 = init_parameters()
    for i in range(epoch):
        # in order to implement momentum technique
        weight_1_change = weight_1 - old_weight_1
        bias_1_change = bias_1 - old_bias_1
        weight_2_change = weight_2 - old_weight_2
        bias_2_change = bias_2 - old_bias_2

        Z1, activation_in, Z2, activation_out = forward_propagation(weight_1, bias_1, weight_2, bias_2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, activation_in, Z2, activation_out, weight_1, weight_2, X, Y)
        weight_1, bias_1, weight_2, bias_2, old_weight_1, old_bias_1, old_weight_2, old_bias_2 = update_parameters(
            weight_1, bias_1, weight_2, bias_2, weight_1_change,
            bias_1_change, weight_2_change, bias_2_change, dW1, db1,
            dW2, db2, learning_rate, momentum)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(activation_out)
            print(f"acc: {get_accuracy(predictions, Y)}")
    return weight_1, bias_1, weight_2, bias_2


# get final results of saved propagation and run eval data through
def make_predictions(X, weight_1, bias_1, weight_2, bias_2):
    _, _, _, activation_out = forward_propagation(weight_1, bias_1, weight_2, bias_2, X)
    predictions = get_predictions(activation_out)
    return predictions


# data preparation(we want to transpose the matrices in order to have examples on columns rather than rows)
data_train = pd.read_csv("mnist_train.csv")
data_eval = pd.read_csv("mnist_test.csv")
data_train = np.array(data_train)
data_eval = np.array(data_eval)

np.random.shuffle(data_train)
np.random.shuffle(data_eval)

data_train = data_train.T
Y_train = data_train[0]
X_train = data_train[1:]

data_eval = data_eval.T
Y_eval = data_eval[0]
X_eval = data_eval[1:]
# normalization
X_train = X_train.astype('float32') / 255
X_eval = X_eval.astype('float32') / 255
# training(acc goes around 85/90%)
W1, b1, W2, b2 = gradient_descent_with_momentum(X_train, Y_train, 150)
# checking validation accuracy
eva_predictions = make_predictions(X_eval, W1, b1, W2, b2)
print(f"eval acc: {get_accuracy(eva_predictions, Y_eval)}")
