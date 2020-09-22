import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def read_data():
    train1 = normalize(np.loadtxt("train1", delimiter="\t"), axis = 0)
    train2 = normalize(np.loadtxt("train2", delimiter="\t"), axis = 0)
    test1 = normalize(np.loadtxt("test1", delimiter="\t"), axis = 0)
    test2 = normalize(np.loadtxt("test2", delimiter="\t"), axis = 0)
    train1_x , train1_y = train1[:, 0].reshape(-1), train1[:, 1].reshape(-1)
    train2_x , train2_y = train2[:, 0].reshape(-1), train2[:, 1].reshape(-1)
    test1_x , test1_y = test1[:, 0].reshape(-1), test1[:, 1].reshape(-1)
    test2_x , test2_y = test2[:, 0].reshape(-1), test2[:, 1].reshape(-1)
    return train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y

def forward(W, x, b):
    return W * x + b

def backward(x, y, y_pred):
    return np.dot((y_pred - y).T, x), np.sum(y_pred - y)

def compute_loss(y_pred, y):
    return 0.5 * np.sum(np.power((y_pred - y), 2))

def plot_network(x, y, W, b):
    plt.scatter(x, y)
    tmp = np.linspace(np.amin(x), np.amax(x), 500)
    plt.plot(tmp, forward(W, tmp, b), color="red")

def get_initial_weights(seed):
    np.random.seed(seed)
    W = np.random.rand() * 0.1
    b = np.random.rand() * 0.1
    return W, b

train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y = read_data()
W, b = get_initial_weights(0)
lr = 8 * 10**-3
n_epochs = 180

x, y = train2_x, train2_y
for i in range(n_epochs):
    y_pred = forward(W, x, b)
    loss = compute_loss(y_pred, y)
    dW, db = backward(x, y, y_pred)
    W -= lr * dW
    b -= lr * db

print("Train loss: {}".format(round(compute_loss(forward(W, x, b), y), 3)))
plot_network(x, y, W, b)

x, y = test2_x, test2_y
y_pred = forward(W, x, b)
test_loss = compute_loss(y_pred, y)
print("Test loss: {}".format(round(test_loss, 2)))
plt.show()