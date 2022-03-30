import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
 

image = Image.open('422.png')
image = image.resize((300, 200))
image = np.array(image)
X = image.reshape(-1, 4)
X = X/255
num_clusters = 30
max_iters = 200
threshold = 0.0001

for i in range(max_iters):
    distances = np.array([])
    if i == 0:
        center_indexs = list(np.random.randint(0, len(X), num_clusters))
        centers = X[center_indexs]
    raw_centers = centers.copy()
    for center in centers:
        distances = np.append(distances, list(map(np.linalg.norm, X - center)))
    distances = distances.reshape(num_clusters, -1)

    cluster_belongings = distances.argmin(axis=0)

    cluster_points = dict()
    for k in range(num_clusters):
        cluster_points_indexs = np.where(cluster_belongings == k)[0]
        new_center =  X[cluster_points_indexs].mean(axis=0)
        centers[k] = new_center
        cluster_points.update({k: np.where(cluster_belongings == k)[0]})
    print(np.sum(raw_centers - centers)**2)
    if np.sum(raw_centers - centers)**2 <= threshold:
        break


new_image = np.zeros((len(X), 4))
for k in range(num_clusters):
    new_image[cluster_points[k], :] = centers[k]


plt.imshow(new_image.reshape(200, 300, 4))
plt.show()