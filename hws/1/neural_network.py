import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math

# np.random.seed(1)

def read_data():
    # train1 = normalize(np.loadtxt("train1", delimiter="\t"), axis = 0)
    # train2 = normalize(np.loadtxt("train2", delimiter="\t"), axis = 0)
    # test1 = normalize(np.loadtxt("test1", delimiter="\t"), axis = 0)
    # test2 = normalize(np.loadtxt("test2", delimiter="\t"), axis = 0)

    train1 = np.loadtxt("train1", delimiter="\t")
    train2 = np.loadtxt("train2", delimiter="\t")
    test1 = np.loadtxt("test1", delimiter="\t")
    test2 = np.loadtxt("test2", delimiter="\t")
    train1 = (train1 - np.mean(train1, axis=0)) / np.std(train1, axis = 0)
    train2 = (train2 - np.mean(train2, axis=0)) / np.std(train2, axis = 0)
    test1 = (test1 - np.mean(test1, axis=0)) / np.std(test1, axis = 0)
    test2 = (test2 - np.mean(test2, axis=0)) / np.std(test2, axis = 0)
    train1_x , train1_y = train1[:, 0].reshape(-1), train1[:, 1].reshape(-1)
    train2_x , train2_y = train2[:, 0].reshape(-1), train2[:, 1].reshape(-1)
    test1_x , test1_y = test1[:, 0].reshape(-1), test1[:, 1].reshape(-1)
    test2_x , test2_y = test2[:, 0].reshape(-1), test2[:, 1].reshape(-1)
    return train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y

def get_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_derivative_sigmoid(x):
    s = get_sigmoid(x)
    return np.multiply(s, 1 - s)

def forward(x, W1, b1, W2, b2):
    activations = np.matmul(W1, x.reshape(1, -1))
    activations += b1[np.newaxis, :][0]
    hidden_units = get_sigmoid(activations)
    y_preds = np.matmul(W2.T, hidden_units) + b2
    return y_preds, activations, hidden_units

def compute_loss(x, y, W1, b1, W2, b2):
    y_preds, _, _ = forward(x, W1, b1, W2, b2)
    return (0.5 * np.sum(np.power(y_preds - y.reshape(1, -1), 2))) / len(x)

def backward(x, y, y_preds, activations, hidden_units, W2):
    num_hidden_units = hidden_units.shape[0]
    dW1, db1, dW2, db2 = np.zeros((num_hidden_units, 1)), np.zeros((num_hidden_units, 1)), np.zeros((num_hidden_units, 1)), 0
    batch_size = len(x)
    tmp = (y_preds - y.reshape(1, -1)).reshape(-1)
    for i in range(batch_size):
        db2 += tmp[i]
        dW2 += tmp[i] * hidden_units[:, i].reshape(-1, 1)
    db2 /= batch_size
    dW2 /= batch_size

    tmp = tmp.reshape(-1, 1) * W2.T
    for i in range(batch_size):
        tmp[i] = np.multiply(get_derivative_sigmoid(activations[:, i]), tmp[i])

    for i in range(batch_size):
        db1 += tmp[i, :].reshape(-1,1)
        dW1 += (tmp[i, :] * x[i]).reshape(-1,1)
    db1 /= batch_size
    dW1 /= batch_size
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, prev_dW1, prev_db1, prev_dW2, prev_db2, momentum, lr):
    delta_W1 = momentum * prev_dW1 + lr * dW1
    delta_b1 = momentum * prev_db1 + lr * db1
    delta_W2 = momentum * prev_dW2 + lr * dW2
    delta_b2 = momentum * prev_db2 + lr * db2
    return W1 - delta_W1, b1 - delta_b1, W2 - delta_W2, b2 - delta_b2, delta_W1, delta_b1, delta_W2, delta_b2

def get_batch(x, y, i, batch_size):
    start = i * batch_size
    end = (i+1) * batch_size

    if end >= len(x):
        end = len(x)
    return x[start:end], y[start:end]

def gradient_descent(x, y, num_hidden_units, lr,  momentum, batch_size, W1, b1, W2, b2):
    num_iterations = math.ceil(len(x) / batch_size)
    prev_dW1, prev_db1, prev_dW2, prev_db2 = 0, 0, 0, 0
    for i in range(num_iterations):
        x_i, y_i = get_batch(x, y, i, batch_size)
        y_preds, activations, hidden_units = forward(x_i, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(x_i, y_i, y_preds, activations, hidden_units, W2)
        W1, b1, W2, b2, prev_dW1, prev_db1, prev_dW2, prev_db2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, prev_dW1, prev_db1, prev_dW2, prev_db2, momentum, lr)
    return W1, b1, W2, b2

def initialize_parameters(shape):
    W1 = np.random.normal(0, 1, (shape, 1))
    W2 = np.random.normal(0, 1, (shape, 1))
    b1 = np.random.normal(0, 1, (shape, 1))
    b2 = np.random.normal(0, 1, (1,1))
    return W1, b1, W2, b2

def save(W1, b1, W2, b2, id):
    np.savetxt('W1_' + str(id) + ".txt", W1, delimiter=',')
    np.savetxt('b1_' + str(id) + ".txt", b1, delimiter=',')
    np.savetxt('W2_' + str(id) + ".txt", W2, delimiter=',')
    np.savetxt('b2_' + str(id) + ".txt", b2, delimiter=',')

def load(id):
    W1 = np.loadtxt('W1_' + str(id) + ".txt", delimiter=',')
    b1 = np.loadtxt('b1_' + str(id) + ".txt", delimiter=',')
    W2 = np.loadtxt('W2_' + str(id) + ".txt", delimiter=',')
    b2 = np.loadtxt('b2_' + str(id) + ".txt", delimiter=',')
    return W1, b1, W2, b2

def train(num_hidden_units, lr, max_epochs, threshold, momentum, batch_size, dr, dataset_id):
    W1, b1, W2, b2 = initialize_parameters(num_hidden_units)
    train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y = read_data()
    if dataset_id == 2:
        x, y = train2_x, train2_y
        x_test, y_test = test2_x, test2_y
    else:
        x, y = train1_x, train1_y
        x_test, y_test = test1_x, test1_y

    train_loss = np.zeros((max_epochs + 1, 1)).reshape(-1)
    epoch = 0

    while True:
        W1, b1, W2, b2 = gradient_descent(x, y, num_hidden_units, lr,  momentum, batch_size, W1, b1, W2, b2)
        train_loss[epoch] = compute_loss(x, y, W1, b1, W2, b2)

        if epoch % 1000 == 999:
            lr *= dr
            print("Epoch, Loss -> {}, {}".format(epoch+1, train_loss[epoch]))

        if (epoch != 1 and abs(train_loss[epoch] - train_loss[epoch - 1]) <= threshold) or epoch == max_epochs:
            break
        epoch += 1
    
    save(np.array(W1), np.array(b1), np.array(W2), np.array(b2), dataset_id)
    print("Train loss: {}".format(round(train_loss[epoch], 4)))
    return x, y, x_test, y_test, epoch, train_loss, W1, b1, W2, b2

def plot_network(x, y, W1, b1, W2, b2):
    plt.scatter(x, y)
    xx = np.linspace(np.amin(x), np.amax(x), 10000)
    y_preds, _, _ = forward(xx, W1, b1, W2, b2)
    plt.plot(xx, y_preds.reshape(-1),  color="red")

def plot_hidden_layer(x, W1, b1, W2, b2):
    xx = np.linspace(np.amin(x), np.amax(x), 10000)
    _, _, hidden_units = forward(xx, W1, b1, W2, b2)
    for i in range(len(W1)):
        plt.plot(xx, hidden_units[i, :])

def plot_losses(epochs, losses):
    plt.plot(list(range(1, epochs+2)), losses[:epochs+1])

def plot_all(x, y, epochs, train_losses, W1, b1, W2, b2):
    plt.subplot(1,3,1)
    plot_network(x, y, W1, b1, W2, b2)
    plt.subplot(1,3,2)
    plot_hidden_layer(x, W1, b1, W2, b2)
    plt.subplot(1,3,3)
    plot_losses(epochs, train_losses)
    plt.show(block = False)
    plt.show()

def plot_all2(x1, y1, x2, y2, epochs, W1, b1, W2, b2, x1_1, y1_1, x2_1, y2_1, epochs_1, W1_1, b1_1, W2_1, b2_1):
    plt.subplot(3,2,1)
    plot_network(x1, y1, W1, b1, W2, b2)
    plt.subplot(3,2,3)
    plot_network(x2, y2, W1, b1, W2, b2)
    plt.subplot(3,2,5)
    plot_hidden_layer(x1, W1, b1, W2, b2)
    plt.subplot(3,2,2)
    plot_network(x1_1, y1_1, W1_1, b1_1, W2_1, b2_1)
    plt.subplot(3,2,4)
    plot_network(x2_1, y2_1, W1_1, b1_1, W2_1, b2_1)
    plt.subplot(3,2,6)
    plot_hidden_layer(x1_1, W1_1, b1_1, W2_1, b2_1)
    plt.show(block = False)
    plt.show()

num_hidden_units, lr, max_epochs, threshold, momentum, batch_size = 8, 0.0075, 250000, 10**-8, 0, 8
x, y, x_test, y_test, epoch, train_loss, W1, b1, W2, b2 = train(num_hidden_units, lr, max_epochs, threshold, momentum, batch_size, 1, 2)
test_loss = compute_loss(x_test, y_test, W1, b1, W2, b2)
print("Test loss: {}".format(round(test_loss, 4)))
plot_network(x, y, W1, b1, W2, b2)
plt.show()