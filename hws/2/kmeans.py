import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

class KMeansClustering:
    def __init__(self, filename, cluster_count):
        self.data = np.loadtxt(filename)
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.cluster_count = cluster_count
        self.cluster_centers = self.data[np.random.choice(self.data.shape[0], size=self.cluster_count), :]

    def cluster(self, epochs = 100):
        next_centers = np.zeros((self.cluster_count, 2))
        prev_centers = np.zeros((self.cluster_count, 2))
        assignments = []

        for epoch in range(epochs):
            assignments = [[] for i in range(self.cluster_count)]

            for i in range(len(self.data)):
                cluster = np.argmin(np.sqrt(np.sum(np.power(self.cluster_centers - self.data[i], 2), axis=1)))
                next_centers[cluster] += self.data[i]
                assignments[cluster].append(i)

            lengths = np.array([len(assignments[j]) for j in range(self.cluster_count)])

            if np.array_equal(np.divide(next_centers, lengths[:, np.newaxis]), prev_centers):
                break
            else:
                prev_centers = np.copy(self.cluster_centers)
                self.cluster_centers = np.divide(next_centers, lengths[:, np.newaxis])
                next_centers = np.zeros((self.cluster_count, 2))


        sse = [] 
        for i in range(len(assignments)):
            i_data = np.reshape(self.data[assignments[i]], (-1,2))
            sse.append(((np.ones((len(i_data), 2)) * self.cluster_centers[i][np.newaxis, :] - i_data) ** 2).sum())
        print(sse)

        figure = plt.figure()
        axis = figure.add_subplot(111)
        
        for i in range(self.cluster_count):
            if len(assignments[i]) > 0:
                t = np.array([self.data[j] for j in assignments[i]])
                axis.scatter(t[:,0], t[:,1], s = 5)
        plt.title("Kmeans Epoch " + str(epoch))        
        plt.show()