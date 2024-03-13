import matplotlib.pyplot as plt
import numpy

data1 = numpy.loadtxt('projekt/alg1_duze.txt')
data2 = numpy.loadtxt('projekt/alg1_srednie.txt')
data3 = numpy.loadtxt('projekt/alg1_male.txt')
data_1 = numpy.loadtxt('projekt/alg2_duze.txt')
data_2 = numpy.loadtxt('projekt/alg2_srednie.txt')
data_3 = numpy.loadtxt('projekt/alg2_male.txt')
sort_indices1 = numpy.argsort(data1[:, 0])
sort_indices2 = numpy.argsort(data2[:, 0])
sort_indices3 = numpy.argsort(data3[:, 0])
sort_indices_1 = numpy.argsort(data_1[:, 0])
sort_indices_2 = numpy.argsort(data_2[:, 0])
sort_indices_3 = numpy.argsort(data_3[:, 0])
sorted_data1 = data1[sort_indices1]
sorted_data2 = data2[sort_indices2]
sorted_data3 = data3[sort_indices3]
sorted_data_1 = data_1[sort_indices_1]
sorted_data_2 = data_2[sort_indices_2]
sorted_data_3 = data_3[sort_indices_3]

x1 = sorted_data1[:, 0]
x2 = sorted_data2[:, 0]
x3 = sorted_data3[:, 0]
x_1 = sorted_data_1[:, 0]
x_2 = sorted_data_2[:, 0]
x_3 = sorted_data_3[:, 0]
y1 = sorted_data1[:, 1]
y2 = sorted_data2[:, 1]
y3 = sorted_data3[:, 1]
y_1 = sorted_data_1[:, 1]
y_2 = sorted_data_2[:, 1]
y_3 = sorted_data_3[:, 1]

plt.subplot(1, 2, 1)
plt.plot(x1, y1, label= 'Duze inputy', color="blue")
plt.plot(x2, y2, label= 'Srednie inputy', color="red")
plt.plot(x3, y3, label= 'Male inputy', color="green")
plt.xlabel('Różnica w sumach podzbiorów')
plt.ylabel('Czas wykonania algorytmu')
plt.ylim(0, 3)
plt.xlim(-20, 1)
plt.title('Algorytm 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_1, y_1, label= 'Duze inputy', color="blue")
plt.plot(x_2, y_2, label= 'Srednie inputy', color="red")
plt.plot(x_3, y_3, label= 'Male inputy', color="green")
plt.xlabel('Różnica w sumach podzbiorów')
plt.ylabel('Czas wykonania algorytmu')
plt.ylim(0, 1)
plt.xlim(-40, 1)
plt.title('Algorytm 2')
plt.legend()
plt.tight_layout()
plt.show()