import numpy
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib as mpl
from libKMCUDA import kmeans_cuda

# numpy.random.seed(0)
# arr = numpy.empty((10000, 2), dtype=numpy.float32)
# arr[:2500] = numpy.random.rand(2500, 2) + [0, 2]
# arr[2500:5000] = numpy.random.rand(2500, 2) - [0, 2]
# arr[5000:7500] = numpy.random.rand(2500, 2) + [2, 0]
# arr[7500:] = numpy.random.rand(2500, 2) - [2, 0]
# centroids, assignments, ave = kmeans_cuda(arr, 4, verbosity=2, average_distance=True, seed=3)
# print("avearge distance:", ave)
# print(centroids)
# plt.scatter(arr[:, 0], arr[:, 1], c=assignments)
# plt.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
# plt.savefig("km_ex1.png")


numpy.random.seed(0)
arr = numpy.empty((10000, 2), dtype=numpy.float32)
for i in range(10000):
    arr[i, :] =  numpy.random.normal(size=2) * 0.5 + numpy.array([i // 1000, i // 1000 + 1])
print(arr[:10, :])
centroids, assignments, avg_distance = kmeans_cuda(
    arr, 10, tolerance=0, metric="l2", verbosity=1, seed=3, average_distance=True)
print("Average distance between centroids and members:", avg_distance)
print(centroids)
plt.figure()
plt.scatter(arr[:, 0], arr[:, 1], c=assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
plt.savefig("km_ex2.png")

# numpy.random.seed(0)
# arr = numpy.empty((10000, 2), dtype=numpy.float32)
# angs = numpy.random.rand(10000) * 2 * numpy.pi
# for i in range(10000):
#     arr[i] = numpy.sin(angs[i]), numpy.cos(angs[i])
# centroids, assignments, avg_distance = kmeans_cuda(
#     arr, 4, metric="l2", verbosity=1, seed=3, average_distance=True)
# print("Average distance between centroids and members:", avg_distance)
# print(centroids)
# plt.figure()
# plt.scatter(arr[:, 0], arr[:, 1], c=assignments)
# plt.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
# plt.savefig("km_ex3.png")
