import time
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from multiprocessing import Process, cpu_count

lena = img.imread("lenna.png")

result = np.zeros(np.shape(lena))

lena_pad = np.pad(lena, 1, 'edge')
kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

y_l, x_l = np.shape(lena_pad)
y_k, x_k = np.shape(kernel)

for ox in range(x_l-x_k+1):
    for oy in range(y_l-y_k+1):
        arr_sum = kernel * lena_pad[ox:ox+x_k, oy:oy+y_k]
        result[ox,oy] = np.sum(arr_sum)

plt.imshow(result, cmap='gray')
plt.savefig('lab02.png')