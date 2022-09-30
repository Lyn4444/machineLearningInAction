import numpy as np
import matplotlib.pyplot as plt

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
fw = open('../testSetRBF2.txt', 'w')

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(100):
    [x, y] = np.random.uniform(0, 1, 2)
    xpt = x * np.cos(2.0 * np.pi * y)
    ypt = x * np.sin(2.0 * np.pi * y)
    if x > 0.5:
        xcord0.append(xpt)
        ycord0.append(ypt)
        label = -1.0
    else:
        xcord1.append(xpt)
        ycord1.append(ypt)
        label = 1.0
        fw.write('%f\t%f\t%f\n' % (xpt, ypt, label))
ax.scatter(xcord0, ycord0, marker='s', s=90)
ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
plt.title('Non-linearly Separable Data for Kernel Method')
plt.show()
fw.close()
