/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 


typedef float typeId;


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(typeId *h_Dst, typeId *h_Src, typeId *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      typeId sum = 0;
	    
      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}

/***************************************************	
 ********** ROW CONVOLUTION GPU	********************
 ***************************************************
 */

__global__ void convolutionRowGPU(typeId *d_Dst, typeId *d_Src, typeId *d_Filter, int imageW, int imageH, int filterR){
  
    int k;
    typeId sum = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y + filterR;
    int col = blockDim.x * blockIdx.x + threadIdx.x + filterR;
    int padW = imageW + filterR*2;
    
    for (k = -filterR; k <= filterR; k++) {
		int d = col + k;

		sum += d_Src[row * padW + d] * d_Filter[filterR - k];
    }
	d_Dst[row * padW + col] = sum;
    
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(typeId *h_Dst, typeId *h_Src, typeId *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      typeId sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

/***********************************************************	
 ******** COLUMN CONVOLUTION GPU ***************************
 ***********************************************************
 */
__global__ void convolutionColumnGPU(typeId *d_Dst, typeId *d_Src, typeId *d_Filter, int imageW, int imageH, int filterR) {
    
  int k;
  typeId sum = 0;
  int row = blockDim.y * blockIdx.y + threadIdx.y + filterR;
  int col = blockDim.x * blockIdx.x + threadIdx.x + filterR;
  int padW = imageW + filterR*2;
  
  for (k = -filterR; k <= filterR; k++) {
	int d = row + k;

	sum += d_Src[col + padW * d] * d_Filter[filterR - k];
  }
  d_Dst[row * padW + col] = sum;
  
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    typeId
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_PaddedInput,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *result,
    diff;

    float elapsedTime;

    cudaSetDevice(0);
    struct timespec  tv1, tv2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int imageW;
    int imageH;
    unsigned int i, j;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    
    h_Filter    = (typeId *)malloc(FILTER_LENGTH * sizeof(typeId));
    h_Input     = (typeId *)malloc(imageW * imageH * sizeof(typeId));
    h_Buffer    = (typeId *)malloc(imageW * imageH * sizeof(typeId));
    h_OutputCPU = (typeId *)malloc(imageW * imageH * sizeof(typeId));
    result 	= (typeId *)malloc((imageW+2*filter_radius) * (imageH+2*filter_radius)* sizeof(typeId));
    h_PaddedInput  = (typeId *)malloc((imageW+filter_radius*2 )*(2*filter_radius+ imageH) * sizeof(typeId));
    // Memory allocation check if any of them not allocated then error
    
    if(!(h_Filter && h_Input && h_Buffer && h_OutputCPU && h_PaddedInput && result)) {
		printf("Error allocating memory\n");
		exit(EXIT_FAILURE);
    }
    
    // Memory allocation on Device
    cudaMalloc(&d_Filter,FILTER_LENGTH*sizeof(typeId));
    cudaMalloc(&d_Input,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));
    cudaMalloc(&d_Buffer,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));
    cudaMalloc(&d_OutputGPU,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));
    
    // Check memory allocation on Device, if any of them failed, exit
    if (!(d_Filter && d_Input && d_Buffer && d_OutputGPU)) {
		printf("Cuda memory allocation failed\n");
		exit(EXIT_FAILURE);
    }
    // Initializing device values
    cudaMemset(d_OutputGPU,0,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));
    cudaMemset(d_Buffer,0,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));
    cudaMemset(d_Input,0,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(typeId));

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (typeId)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (typeId)rand() / ((typeId)RAND_MAX / 255) + (typeId)rand() / (typeId)RAND_MAX;
    }


    // CPU computation
    printf("CPU computation...\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); 
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);     
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
     
    printf ("CPU time:  %10g seconds\n\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec)); 
 
	dim3 dimGrid((imageH-1)/32 + 1,(imageH-1)/32 + 1);
	dim3 dimBlock(32,32);
    
    // init padded Input
    for(i=0;i<(imageW+2*filter_radius)*(imageW+2*filter_radius);i++) h_PaddedInput[i]=0;
    // filling the cells   
    for(i=0;i<imageH;i++){
      for(j=0;j<imageW;j++){
		h_PaddedInput[(i+filter_radius)*(2*filter_radius+imageW)+filter_radius+j]=h_Input[i*imageW+j];
      }
    }
    
    printf("GPU computation... \n");
    
    cudaMemcpy(d_Filter,h_Filter,FILTER_LENGTH*sizeof(typeId),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input,h_PaddedInput,(imageH+2*filter_radius)*(imageW+2*filter_radius)*sizeof(typeId),cudaMemcpyHostToDevice);
	cudaEventRecord(start,0);
    // kernel invocation
    convolutionRowGPU <<< dimGrid,dimBlock >>>(d_Buffer,d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaThreadSynchronize();
    
    //Check for errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
    	printf("Device Error:%s\n",cudaGetErrorString(err));
    	cudaDeviceReset();
    	return 0;
    }
    
    convolutionColumnGPU <<< dimGrid,dimBlock >>>(d_OutputGPU,d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaThreadSynchronize();
    
    //Check for errors
    err = cudaGetLastError();
    if(err != cudaSuccess) {
		printf("Device Error:%s\n",cudaGetErrorString(err));
		cudaDeviceReset();
		return 0;
	}
	cudaEventRecord(stop,0);
    //Copy results to host
    cudaMemcpy(result, d_OutputGPU, (imageH+2*filter_radius)*(imageW+2*filter_radius)*sizeof(typeId), cudaMemcpyDeviceToHost);
    
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("GPU time: %f ms.\n\n",elapsedTime);
    
    // Checking accuracy error CPU vs CPU
    for(i=0; i<imageW; i++) {
		for(j=0; j<imageH; j++){
			diff= h_OutputCPU[i*imageW+j]-result[(i+filter_radius)*(imageW+2*filter_radius)+filter_radius+j];
				if(ABS(diff) > accuracy) {
					printf("Accuracy error <<%f>>\n ",ABS(diff));
					free(h_OutputCPU);
					free(h_Buffer);
					free(h_Input);
					free(h_Filter);
					free(h_PaddedInput);
					cudaFree(d_OutputGPU);
					cudaFree(d_Buffer);
					cudaFree(d_Input);
					cudaFree(d_Filter);
					cudaDeviceReset();
					exit(EXIT_FAILURE);
				}
		}
	}

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(h_PaddedInput);
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
	cudaFree(d_Filter);
    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
