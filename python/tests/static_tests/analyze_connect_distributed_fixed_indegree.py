import numpy as np

data1 = np.loadtxt("tmp1.dat")
data2 = np.loadtxt("tmp2.dat")

mean1 = data1.mean(axis=0)
mean2 = data2.mean(axis=0)
stdm1 = data1.std(axis=0) / np.sqrt(data1.shape[0])
stdm2 = data2.std(axis=0) / np.sqrt(data2.shape[0])
print(f"N. of samples: {data1.shape[0]}")
print("Indegree per host computed from simulations:")
print(f"{mean1} +- {stdm1}")
print(f"{mean2} +- {stdm2}")

source_size = np.asarray([4, 3, 3])
target_size = [3, 4]
indegree = 4

theor1 = target_size[0] * indegree * source_size / sum(source_size)
theor2 = target_size[1] * indegree * source_size / sum(source_size)
print("Theoretical:")
print(theor1)
print(theor2)
