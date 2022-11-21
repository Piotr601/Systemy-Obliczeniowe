import time
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def first_ex(kernel, lena_pad):

    y_l, x_l = np.shape(lena_pad)
    y_k, x_k = np.shape(kernel)

    t_0f = time.time()
    
    for ox in range(x_l-x_k+1):
        for oy in range(y_l-y_k+1):
            arr_sum = kernel * lena_pad[ox:ox+x_k, oy:oy+y_k]
            result[ox,oy] = np.sum(arr_sum)

    print("Execution time: %6f sec" % (time.time()-t_0f)) 
    
    plt.imshow(result, cmap='gray')
    plt.savefig('lab02.png')
    
    
def second_ex(ox ,result_sec, kernel, lena_pad):
    y_l, x_l = np.shape(lena_pad)
    y_k, x_k = np.shape(kernel)
    
    for oy in range(y_l-y_k+1):
        arr_sum = kernel * lena_pad[ox:ox+x_k, oy:oy+y_k]
        result_sec[ox,oy] = np.sum(arr_sum)
        
    return result_sec.astype(int)

    
def main(result_sec, kernel, lena_pad):
    y_l, x_l = np.shape(lena_pad)
    y_k, x_k = np.shape(kernel)
    
    with ProcessPoolExecutor(max_workers = 5) as executor:
        ress = zip(result_sec, executor.map(second_ex, range(x_l-x_k+1), result_sec, kernel, lena_pad))
        
    return ress


if __name__ == '__main__':
    lena = img.imread("lenna.png")
    result = np.zeros(np.shape(lena))
    kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])
    lena_pad = np.pad(lena, 1, 'edge')
    
    # To collect executing time the first solution
    first_ex(kernel, lena_pad)
    
    result_sec = np.zeros(np.shape(lena))
    
    # Going per row
    t_1f = time.time()
    
    res = main(result_sec, kernel, lena_pad)
    
    print("Execution time: %6f sec" % (time.time()-t_1f)) 
    
    plt.imshow(res, cmap='gray')
    plt.savefig('lab02a.png')
    