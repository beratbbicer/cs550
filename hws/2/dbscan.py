import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

class DBSCANClustering:
    def __init__(self, filename, eps, min_points):
        self.data = np.loadtxt(filename)
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = np.take(self.data,np.random.permutation(self.data.shape[0]),axis=0)
        self.eps = eps
        self.minpts = min_points
        self.labels = -np.ones(len(self.data)) # undefined by default
        self.cno = 0

    def range_query(self, point):
        points = []
        labels = []
        indices = []
        for i in range(len(self.data)):
            if np.linalg.norm(point - self.data[i]) <= self.eps and np.array_equal(self.data[i], point) == False:
                points.append(self.data[i])
                labels.append(self.labels[i])
                indices.append(i)
        return np.array(points), np.array(labels, dtype = np.int32), np.array(indices, dtype = np.int32)

    def cluster(self):
        for i in range(len(self.data)):
            if self.labels[i] != -1:
                continue

            npoints, nlabels, nindices = self.range_query(self.data[i])
            if len(npoints) < self.minpts:
                self.labels[i] = 0 # noise
                continue

            self.cno += 1
            self.labels[i] = self.cno
            count = -1
            while True:
                count += 1
                if count >= len(npoints):
                    break

                # print("count: " + str(count))
                
                if nlabels[count] == 0:
                    # print("0: " + str(nlabels[count]) + ", " + str(count))
                    self.labels[nindices[count]] = self.cno
                    """elif nlabels[count] != -1:
                    # print("-1: " + str(nlabels[count]) + ", " + str(count))
                    continue"""
                else:
                
                    # print("else: " + str(nlabels[count]) + ", " + str(count))
                    self.labels[nindices[count]] = self.cno
                    npoints2, nlabels2, nindices2 = self.range_query(npoints[count])

                    if len(npoints2) >= self.minpts:
                        # print("new_appends: " + str(np.append(npoints, npoints2, axis=0).shape) + ", " + str(count))
                        self.labels[nindices2] = self.cno

                        for j in range(len(npoints2)):
                            if npoints2[j] not in npoints:
                                self.labels[nindices2[j]] = self.cno
                                npoints = np.vstack((npoints, npoints2[j]))
                                nlabels = np.append(nlabels, nlabels2[j])
                                nindices = np.append(nindices, nindices2[j])                
        sse = [] 
        for i in range(len(np.unique(self.labels))):
            i_data = np.reshape(self.data[np.argwhere(self.labels == i)], (-1,2))
            mean_i = np.mean(i_data, axis = 0)
            sse.append(((np.ones((len(i_data), 2)) * mean_i[np.newaxis, :] - i_data) ** 2).sum())
        print(sse)

        figure = plt.figure()
        axis = figure.add_subplot(111)        
        for i in range(len(np.unique(self.labels))):
            t = np.reshape(self.data[np.argwhere(self.labels == np.unique(self.labels)[i])], (-1,2))
            axis.scatter(t[:,0], t[:,1], s = 5)
        plt.title("DBSCAN")
        plt.show()