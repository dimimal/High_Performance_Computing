## Comparison of various convolution algorithms with CUDA
## Tiled vs Non-tiled Version

make 

./exec_non_tiled

'''
Enter filter radius : 32
Enter image size. Should be a power of two and greater than 65 : 6400
Image Width x Height = 6400 x 6400

Allocating and initializing host arrays...
CPU computation...
CPU time: 7.05165 seconds

GPU computation...
GPU time: 141.062302 ms.

Accuracy error <<1.000000>>
'''