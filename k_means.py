"""K-means clustering toy implementation
Author: Saimon
Email: saimoncse19@gmail.com
Date: March 09, 2019 """


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use("ggplot")

# array 'a' is our toy input data
a = np.array([[2, 4], [2, 3], [5, 2], [6, 2], [5, 2.5], [2.5, 3.5]])

# Assuming k=2, we get the following two clusters
# Initially, empty lists are assigned to them
c1 = []
c2 = []

# garbage lists to keep track of centroid updates
garbage1 = []
garbage2 = []

# as k = 2, randomly choosing centroid_1 and centroid 2 from our toy input data
centroid_1 = np.array([[2, 4]])
centroid_2 = np.array([[5, 2]])

garbage1.append(list(centroid_1))
garbage2.append(list(centroid_2))

for i in a:
    distance1 = np.linalg.norm(i - centroid_1)      # euclidean distance between i and centroid_1
    distance2 = np.linalg.norm(i - centroid_2)      # euclidean distance between i and centroid_2
    if distance1 < distance2:
        c1.append(i)
        centroid_1 = np.array(c1).mean(axis=0)
        garbage1.append(list(centroid_1))
    else:
        c2.append(i)
        centroid_2 = np.array(c2).mean(axis=0)
        garbage2.append(list(centroid_2))

# converting c1 and c2 into numpy arrays
cluster1 = np.array(c1)
cluster2 = np.array(c2)
plt.title("K-means clustering toy Implementation")
plt.xlabel("X axis")
plt.ylabel("Y axis")

# Plotting the cluster1 and cluster2
plt.scatter(cluster1[:, 0], cluster1[:, 1], label = "cluster1", color="red")
plt.scatter(cluster2[:, 0], cluster2[:, 1], label = "cluster2", color="green")

# Getting the final value of centroid_1 and centroid_2
final_centroid_1 = garbage1[-1]
final_centroid_2 = garbage2[-1]

# Plotting the final values of the centroids
plt.scatter(final_centroid_1[0], final_centroid_1[1], label = "Final Centroid", color="black")
plt.scatter(final_centroid_2[0], final_centroid_2[1], color="black")

plt.legend()
plt.show()
