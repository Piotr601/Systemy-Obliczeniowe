import Z3x
import time
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1,3, figsize=(12,5))

lena = plt.imread('Lab03/lenna.png')
img = ((lena - np.min(lena)) / (np.max(lena) - np.min(lena)) * 255).astype(int)

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype='int')

timer0 = time.time()
after_naive_convolve = Z3x.naive_convolve(img, kernel)
timer0_k = (time.time()-timer0)
print("Naive convolve: %.6f sec" % (timer0_k)) 

timer1 = time.time()
after_speed_convolve = Z3x.speed_convolve(img, kernel)
timer1_k = (time.time()-timer1)
print("Naive convolve: %.6f sec" % (timer1_k)) 

after_speed_convolve = after_speed_convolve[1:513, 1:513]
after_naive_convolve = after_naive_convolve[1:513, 1:513]

ax[0].imshow(img, cmap='binary_r')
ax[0].set_title('Original')
ax[1].imshow(after_speed_convolve, cmap='binary_r')
ax[1].set_title('Speed convolve\n%.6f sec' % timer1_k)
ax[2].imshow(after_naive_convolve, cmap='binary_r')
ax[2].set_title('Naive convolve\n%.6f sec' % timer0_k)

#print(np.shape(img), np.shape(after_naive_convolve), np.shape(after_speed_convolve))
print("Speed convolve is %.1f times faster than naive convolve" % (timer0_k/timer1_k))

plt.tight_layout()
plt.savefig('Lab03/lab03.png')