# High Performance Computing  with CUDA
 Tiled and Non tiled convolution algorithms comparison in CUDA and histogram equalization technique for image processing. 
## Convolutions

### Tiled vs Non-tiled Version

None tiled Version

`make` 
`./exec_non_tiled`

Output:
```
Enter filter radius : 32
Enter image size. Should be a power of two and greater than 65 : 6400
Image Width x Height = 6400 x 6400

Allocating and initializing host arrays...
CPU computation...
CPU time: 7.05165 seconds

GPU computation...
GPU time: 141.062302 ms.
```

Tiled Convolution 

`./tiled_conv`

```
Enter image size. Should be a power of two and greater than 33 : 6400
Image Width x Height = 6400 x 6400

Allocating and initializing host arrays...
CPU computation...
CPU time =    3.54343 seconds
GPU computation...
GPU time: 58.340351 ms.
```

The results have been taken on a different machine than the results above but they are proportional.

<img src="https://github.com/dimimal/High_Performance_Computing/blob/master/Convolutions/Results.png" width="512" height="648" />

## Histogram Equalization
In Histogram Equalization folder run `make` to create the object file.

Pass an input image to equalize along with its output file you want to save into
 
`./exec input.pgm output.pgm`

Results:

<img src="https://drive.google.com/file/d/1NULquE3A6Ay7ZmkGMZtH7w_vjQmURA4P/view?usp=sharing" width="1024" height="768" />

<img src="https://github.com/dimimal/High_Performance_Computing/blob/master/Histogram_Equalization/images/processedXray.pgm?raw=true" width="1024" height="768" />
