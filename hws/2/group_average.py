import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import time

alot = 999999999

class GroupAverageClustering:
    def __init__(self, filename, cluster_count):
        self.data = np.loadtxt(filename)
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = np.take(self.data,np.random.permutation(self.data.shape[0]),axis=0)
        self.distance_matrix = distance.cdist(self.data, self.data, "euclidean")
        self.distance_matrix[self.distance_matrix == 0] = alot
        self.desired_cluster_count = cluster_count
        self.assignments = np.arange(len(self.data))

    def cluster(self):
        epochs = 0
        while True:   
            (x, y) = np.unravel_index(np.argmin(self.distance_matrix, axis=None), self.distance_matrix.shape)
            self.assignments[np.argwhere(self.assignments == y)] = x
            self.distance_matrix[y,:] = alot
            self.distance_matrix[:,y] = alot
            x_data = np.reshape(self.data[np.argwhere(self.assignments == x)], (-1,2))
            for i in range(self.distance_matrix.shape[0]):
                if self.distance_matrix[x, i] != alot:
                    tmp = np.mean(distance.cdist(x_data, np.reshape(self.data[np.argwhere(self.assignments == i)], (-1,2)), "euclidean"))
                    self.distance_matrix[x, i] = tmp # min(tmp, self.distance_matrix[x, i])
                    self.distance_matrix[i, x] = tmp # min(tmp, self.distance_matrix[i, x])

            if len(np.unique(self.assignments)) <= self.desired_cluster_count:
                break
            
            # print("Epoch " + str(epochs))
            epochs += 1

        sse = []
        clusters = np.unique(self.assignments)
        for i in range(len(np.unique(self.assignments))):
            i_data = np.reshape(self.data[np.argwhere(self.assignments == clusters[i])], (-1,2))
            mean_i = np.mean(i_data, axis = 0)
            sse.append(((np.ones((len(i_data), 2)) * mean_i[np.newaxis, :] - i_data) ** 2).sum())
        print(sse)

        figure = plt.figure()
        axis = figure.add_subplot(111)        
        for i in range(self.desired_cluster_count):
            t = np.reshape(self.data[np.argwhere(self.assignments == np.unique(self.assignments)[i])], (-1,2))
            axis.scatter(t[:,0], t[:,1], s = 5)
        plt.title("Group Average")
        plt.show()