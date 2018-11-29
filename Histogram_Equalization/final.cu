#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#include "contrast-enhancement.c"
#include "histogram-equalization.c"

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, unsigned char * h_result);

__global__ void histogram_kernel(int *d_histOut, unsigned char *d_Input, int imgW, int imgH)
{
	
	__shared__ unsigned int temp[256];
	temp[threadIdx.x]=0;
	__syncthreads();
	
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = tx + ty * imgW;
	
	
	while(tx < imgW && ty < imgH)
    {
		atomicAdd( & (temp[d_Input[idx]]) ,1);
		tx += blockDim.x*gridDim.x;
		idx += blockDim.x*gridDim.x;
	}
	__syncthreads();
	
	atomicAdd(&(d_histOut[threadIdx.x]),temp[threadIdx.x]);
	
	
}
__global__ void histogram_equ_kernel(unsigned char * d_Output, unsigned char * d_Input, int imgW, int imgH, int * d_lut )
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = tx + ty * imgW;
	
	__shared__ unsigned int lut[256];
	lut[ threadIdx.x ] = d_lut[ threadIdx.x ];
	__syncthreads();
	
	while(tx < imgW && ty < imgH)
    {
		d_Output[idx] = lut[d_Input[idx]] > 255 ? 255 : (unsigned char) lut[d_Input[idx]];
		tx += blockDim.x*gridDim.x;
		idx += blockDim.x*gridDim.x;
	}
}




void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, unsigned char *h_result)
{
    unsigned int timer = 0, i, j;
    struct timespec  tv1, tv2;
    PGM_IMG img_obuf;
    PGM_IMG d_out;
    d_out.img = h_result;
    d_out.h = img_in.h;
    d_out.w = img_in.w;
    
    printf("Starting CPU processing...\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    img_obuf = contrast_enhancement_g(img_in);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("CPU time = %10g seconds\n",
            (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
            (double) (tv2.tv_sec - tv1.tv_sec));
    
    for( i=0;i<img_obuf.h;i++)
    {
        for(j=0;j<img_obuf.w;j++)
        {
            if(h_result[i*img_obuf.w+j]!=img_obuf.img[i*img_obuf.w+j])
            {
                printf("DIFFERENCE\n %c | %c",h_result[i*img_obuf.w+j],img_obuf.img[i*img_obuf.w+j]);
                free_pgm(d_out);
                //cudaDeviceReset();
                return ;
            }
        }
    }
    write_pgm(d_out, out_filename);
    free_pgm(d_out);
}


PGM_IMG read_pgm(const char * path)
{
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL)
    {
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path)
{
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}


int main(int argc, char *argv[])
{
    cudaSetDevice(0);
    PGM_IMG img_ibuf_g;
	float elapsedTime;
	int size,blocks,*d_histOut,*h_histOutput,*h_lut,*d_lut,cdf,min,d,i;
	cudaEvent_t start, stop;
    unsigned char *d_Input, *d_Output, *h_result;
	
	//struct timespec  tv1, tv2;
	if (argc != 3) 
    {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
	size = img_ibuf_g.w * img_ibuf_g.h;
    
	cudaMalloc(&d_Input,size*sizeof(unsigned char));
	cudaMalloc(&d_histOut,256*sizeof(int));
	cudaMalloc(&d_lut,256*sizeof(int));
	cudaMalloc(&d_Output,size*sizeof(unsigned char));
    
    // Check
    if(!(d_Input && d_histOut && d_lut && d_Output)) 
    {
	  printf("Cuda memory allocation failed\n");
	  exit(EXIT_FAILURE);
      cudaDeviceReset();
	}
	
	cudaMemset(d_histOut,0,256*sizeof(int));
	h_result=(unsigned char *)malloc(size * sizeof(unsigned char));
	h_histOutput=(int *)malloc(256*sizeof(int));
	h_lut= (int *)malloc(sizeof(int)*256);
	
    //check mem allocation
	if(!(h_result && h_histOutput && h_lut))
    {
		printf("Memory allocation failed\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);	
	}

	// define grid and block dimensions
	blocks = img_ibuf_g.w%(8*256)==0 ? img_ibuf_g.w/(8*256) : blocks=img_ibuf_g.w/(8*256)+1;
	
	
	dim3 grid(blocks,img_ibuf_g.h);
	dim3 block(256,1);

	/*************** GPU COMPUTATION ABOVE ****************************************/
   	 cudaEventCreate(&start);
    	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	cudaMemcpy(d_Input,img_ibuf_g.img,size * sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU image copy time :%f ms.\n",elapsedTime);
	cudaEventRecord(start, 0);
	histogram_kernel <<<grid,block>>> ( d_histOut , d_Input , img_ibuf_g.w , img_ibuf_g.h );
	cudaThreadSynchronize();
    	cudaError_t error = cudaGetLastError();
    	if(error!=cudaSuccess)
        {
		printf("Cuda Error: %s\n",cudaGetErrorString(error));
		cudaDeviceReset();
      		exit(EXIT_FAILURE);
        }

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU histogram time :%f ms.\n",elapsedTime);
	cudaEventRecord(start,0);
	cudaMemcpy(h_histOutput,d_histOut,256*sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU histogram copy time :%f ms.\n",elapsedTime);
	cudaEventRecord(start,0);
	
    // Cpu code
	cdf = 0;
   	 min = 0;
	i = 0;
	while(min == 0)
    {
        min = h_histOutput[i++];
    }
   	d = size - min;
    for(i = 0; i <256; i ++)
    {
        cdf += h_histOutput[i];
        h_lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(h_lut[i] < 0)
        {
            h_lut[i] = 0;
    	}
    }

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU lut time :%f ms.\n",elapsedTime);
	cudaEventRecord(start,0);
	cudaMemcpy(d_lut,h_lut,256*sizeof(int),cudaMemcpyHostToDevice);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU lut copy time :%f ms.\n",elapsedTime);
	cudaEventRecord(start,0);
	histogram_equ_kernel<<<grid,block>>>( d_Output, d_Input , img_ibuf_g.w , img_ibuf_g.h , d_lut);
	cudaThreadSynchronize();
    error = cudaGetLastError();
    
    if(error!=cudaSuccess)
    {
      printf("Cuda Error:%s\n",cudaGetErrorString(error));
      cudaDeviceReset();
      exit(EXIT_FAILURE);  
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU histogram-equalization time :%f ms.\n",elapsedTime);
	cudaEventRecord(start,0);
	cudaMemcpy(h_result,d_Output,size * sizeof(unsigned char),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("GPU image copy out time :%f ms.\n",elapsedTime);
	
    /*************************************************************************** *******  FREE MEMMORY AND RUN GRAY_TEST ************************************************************************************/  
    run_cpu_gray_test(img_ibuf_g, argv[2], h_result);
    free_pgm(img_ibuf_g);
	free(  h_histOutput );
	free( h_lut );
	cudaEventDestroy( start);
	cudaEventDestroy( stop );
	cudaFree( d_Input );
	cudaFree( d_Output );
	cudaFree( d_histOut );
	cudaFree( d_lut );
    cudaDeviceReset();
	return 0;
}