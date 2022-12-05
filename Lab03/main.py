import convolve
import time
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1,3, figsize=(12,4))

lena = plt.imread('Lab03/lenna.png')

img = ((lena - np.min(lena)) / (np.max(lena) - np.min(lena)) * 255).astype(int)

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype='int')

timer0 = time.time()
after_naive_convolve = convolve.naive_convolve(img, kernel)
print("Naive convolve: %6f sec" % (time.time()-timer0)) 

timer1 = time.time()
after_speed_convolve = convolve.speed_convolve(img, kernel)
print("Naive convolve: %6f sec" % (time.time()-timer1)) 

ax[0].imshow(img, cmap='binary_r')
ax[0].set_title('Original')
ax[1].imshow(after_speed_convolve, cmap='binary_r')
ax[1].set_title('Speed convolve')
ax[2].imshow(after_naive_convolve, cmap='binary_r')
ax[2].set_title('Naive convolve')

plt.savefig('Lab03/lab03.png')