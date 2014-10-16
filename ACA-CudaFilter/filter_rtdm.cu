// 
// Filters
//

// Includes: system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sys/io.h>

#include <cutil_inline.h>

unsigned int time_GPU = 0;
unsigned int time_CPU = 0;
unsigned int time_mem = 0;
unsigned int time_total = 0;

// Includes: local
#include "bmp.h"

enum {SOBEL_FILTER=1, SOBEL_FILTER5, AVERAGE_FILTER, HIGH_BOOST_FILTER};

#define CLAMP_8bit(x) max(0, min(255, (x)))

char *BMPInFile = "lena.bmp";
char *BMPOutFile = "output.bmp";
char *Filter = "sobel";
int FilterMode  = SOBEL_FILTER;

// Functions
void Cleanup(void);
void ParseArguments(int, char**);
void FilterWrapper(unsigned char* pImageIn, int Width, int Height);

// Kernels
__global__ void SobelFilter(unsigned char *g_DataIn, unsigned char *g_DataOut, int width, int height);
__global__ void AverageFilter(unsigned char *g_DataIn, unsigned char *g_DataOut, int width, int height);
__global__ void SobelFilter5(unsigned char *g_DataIn, unsigned char *g_DataOut, int width, int height, float *sobel_matrix);
__global__ void HighBoostFilter(unsigned char *g_DataIn, unsigned char *g_DataOut, int width, int height);

/* Device Memory */
unsigned char *d_In;
unsigned char *d_Out;
float *sobel_matrix;

// Setup for kernel size
const int TILE_WIDTH    = 6;
const int TILE_HEIGHT   = 6;

const int FILTER_RADIUS = 1;
//const int FILTER_RADIUS = 2; // for sobel 5x5

const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA   = FILTER_DIAMETER * FILTER_DIAMETER;

const int BLOCK_WIDTH   = TILE_WIDTH + 2*FILTER_RADIUS;
const int BLOCK_HEIGHT  = TILE_HEIGHT + 2*FILTER_RADIUS;

const int EDGE_VALUE_THRESHOLD = 70;
//const int EDGE_VALUE_THRESHOLD = 500; //sobel 5
const int HIGH_BOOST_FACTOR = 10;

#include "filter_kernel.cu"

void BitMapRead(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char **data, unsigned char **palete)
{
   size_t palete_size;
   int fd;

   if((fd = open(file, O_RDONLY )) < 0)
           FATAL("Open Source");

   if(read(fd, bmp, BMP_SIZE) != BMP_SIZE)
           FATAL("Read BMP Header");

   if(read(fd, dib, DIB_SIZE) != DIB_SIZE)
           FATAL("Read DIB Header");

   assert(dib->bpp == 8);

   palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
   if(palete_size > 0) {
           *palete = (unsigned char *)malloc(palete_size);
           int go = read(fd, *palete, palete_size);
           if (go != palete_size) {
                   FATAL("Read Palete");
           }
   }

   *data = (unsigned char *)malloc(dib->image_size);
   if(read(fd, *data, dib->image_size) != dib->image_size)
           FATAL("Read Image");

   close(fd);
}


void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char *data, unsigned char *palete)
{
   size_t palete_size;
   int fd;

   palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;

   if((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC,
                             S_IRUSR | S_IWUSR |S_IRGRP)) < 0)
           FATAL("Open Destination");

   if(write(fd, bmp, BMP_SIZE) != BMP_SIZE)
           FATAL("Write BMP Header");

   if(write(fd, dib, DIB_SIZE) != DIB_SIZE)
           FATAL("Write BMP Header");

   if(palete_size != 0) {
           if(write(fd, palete, palete_size) != palete_size)
                   FATAL("Write Palete");
   }
   if(write(fd, data, dib->image_size) != dib->image_size)
           FATAL("Write Image");
   close(fd);
}



void CPU_Sobel(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  const float SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
 // const float SobelMatrix[25] = { -1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1 }; // sobel 5
  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
	imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 9 values to calculate both the direction x and direction y
       float sumX = 0, sumY=0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel * SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
             sumY += Pixel * SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
          }
	}
       imageOut[i*width + j] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0;
    }
  }
}

void CPU_Average(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  
  const float SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
//  const float SobelMatrix[25] = { -1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1 }; // sobel 5
  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
	imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {
	//SUM = 0;
       // sum up the 9 values to calculate both the direction x and direction y
       float sumX = 0, sumY=0, SUM = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(imageIn[i*width + j +  (dy * width + dx)]);
             SUM = SUM + Pixel;
          }
	}
       imageOut[i*width + j] = SUM/9;
	   //SUM = 0.0;
    }
  }
}

void CPU_Boost(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  //float SUM = 0.0;
  int SUM = 0;
  const float SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
//  const float SobelMatrix[25] = { -1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1 }; // sobel 5
  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
	imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {
	//float SUM = 0.0;
	//SUM = 0;
       // sum up the 9 values to calculate both the direction x and direction y
       float sumX = 0, sumY=0;
	   SUM = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(imageIn[i*width + j +  (dy * width + dx)]);
             SUM = SUM + Pixel;
          }
	}
		SUM = SUM/9;
        float px = (float)imageIn[i*width + j];
		imageOut[i*width + j] = CLAMP_8bit((int)(px + HIGH_BOOST_FACTOR*(px-SUM)));
		//SUM = 0.0;
    }
  }
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    struct bmp_header bmp;
    struct dib_header dib;
	
	/*if(FilterMode == SOBEL_FILTER5){
	
		FILTER_RADIUS = 2;
		FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
		FILTER_AREA = FILTER_DIAMETER * FILTER_DIAMETER;
		BLOCK_WIDTH = TILE_WIDTH + 2*FILTER_RADIUS;
		BLOCK_HEIGHT = TILE_HEIGHT + 2*FILTER_RADIUS;
		
	}
	*/
	 // Create the timers
     cutilCheckError(cutCreateTimer(&time_mem));
     cutilCheckError(cutCreateTimer(&time_total));
     cutilCheckError(cutCreateTimer(&time_GPU));
     cutilCheckError(cutCreateTimer(&time_CPU));
	
	float s_SobelMatrix[25];  
	s_SobelMatrix[0] = 1;
	s_SobelMatrix[1] = 2;
	s_SobelMatrix[2]= 0;
	s_SobelMatrix[3] = -2;
	s_SobelMatrix[4] = -1;
	s_SobelMatrix[5] = 4;
	s_SobelMatrix[6] = 8;
	s_SobelMatrix[7] = 0;
	s_SobelMatrix[8] = -8;
	s_SobelMatrix[9] = -4;
	s_SobelMatrix[10] = 6;
	s_SobelMatrix[11] = 12;
	s_SobelMatrix[12] = 0;
	s_SobelMatrix[13] = -12;
	s_SobelMatrix[14] = -6;
	s_SobelMatrix[15] = 4;
	s_SobelMatrix[16] = 8; 
	s_SobelMatrix[17] = 0;
	s_SobelMatrix[18] = -8;
	s_SobelMatrix[19] =-4;
	s_SobelMatrix[20] =1;
	s_SobelMatrix[21] =2;
	s_SobelMatrix[22] =0;
	s_SobelMatrix[23] =-2;
	s_SobelMatrix[24] =-1;
	
    unsigned char *palete = NULL;
    unsigned char *data = NULL, *out = NULL;

    printf("Running %s filter\n", Filter);
    BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
    out = (unsigned char *)malloc(dib.image_size);

    printf("Computing the CPU output\n");
    printf("Image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);
    
	cutilCheckError(cutStartTimer(time_CPU));
	if(FilterMode == SOBEL_FILTER || FilterMode == SOBEL_FILTER5){
	CPU_Sobel(data, out, dib.width, dib.height);
	printf("Running Sobel\n");
	}
	else if(FilterMode == AVERAGE_FILTER){
	CPU_Average(data, out, dib.width, dib.height);
	printf("Running Average\n");
	}
	else if(FilterMode == HIGH_BOOST_FILTER){
	CPU_Boost(data, out, dib.width, dib.height);
	printf("Running boost\n");
	}
	cutilCheckError(cutStopTimer(time_CPU));
	if(FilterMode == SOBEL_FILTER || FilterMode == SOBEL_FILTER5)
    BitMapWrite("CPU_sobel.bmp", &bmp, &dib, out, palete);
	
	else if(FilterMode == AVERAGE_FILTER)
	BitMapWrite("CPU_average.bmp", &bmp, &dib, out, palete);
	
	else if(FilterMode == HIGH_BOOST_FILTER)
	BitMapWrite("CPU_boost.bmp", &bmp, &dib, out, palete);
	
    printf("Done with CPU output\n");
	printf("CPU execution time %f \n", cutGetTimerValue(time_CPU));
	
	
    printf("Allocating %d bytes for image \n", dib.image_size);
	
    cutilSafeCall( cudaMalloc( (void **)&d_In, dib.image_size*sizeof(unsigned char)) );
    cutilSafeCall( cudaMalloc( (void **)&d_Out, dib.image_size*sizeof(unsigned char)) );
    
	// creating space for filter matrix
	cutilSafeCall( cudaMalloc( (void **)&sobel_matrix, 25*sizeof(float)) );
	
	cutilCheckError(cutStartTimer(time_mem));
	
	cudaMemcpy(d_In, data, dib.image_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	cudaMemcpy(sobel_matrix, s_SobelMatrix, 25*sizeof(float), cudaMemcpyHostToDevice);
	
	cutilCheckError(cutStopTimer(time_mem));
    
	FilterWrapper(data, dib.width, dib.height);

    // Copy image back to host
	
	cutilCheckError(cutStartTimer(time_mem));
    cudaMemcpy(out, d_Out, dib.image_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cutilCheckError(cutStopTimer(time_mem));
	
	printf("GPU execution time %f Memtime %f \n", cutGetTimerValue(time_GPU), cutGetTimerValue(time_mem));
    printf("Total GPU = %f \n", (cutGetTimerValue(time_GPU) + cutGetTimerValue(time_mem)));
	// Write output image   
    BitMapWrite(BMPOutFile, &bmp, &dib, out, palete);

    Cleanup();
}

void Cleanup(void)
{
    cutilSafeCall( cudaThreadExit() );
    exit(0);
}


void FilterWrapper(unsigned char* pImageIn, int Width, int Height)
{
   // Design grid disection around tile size
   int gridWidth  = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
   int gridHeight = (Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
   dim3 dimGrid(gridWidth, gridHeight);

   // But actually invoke larger blocks to take care of surrounding shared memory
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

   switch(FilterMode) {
     case SOBEL_FILTER:
     printf("Sobel Filter \n");
	 cutilCheckError(cutStartTimer(time_GPU));
     SobelFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
	 cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	 cutilCheckError(cutStopTimer(time_GPU));
     break;
	 
	 case SOBEL_FILTER5:
	 //FILTER_RADIUS = 2;
     printf("Sobel Filter - 5 \n");
	 cutilCheckError(cutStartTimer(time_GPU));
     SobelFilter5<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height, sobel_matrix);
     cutilCheckMsg("kernel launch failure");
	 cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	 cutilCheckError(cutStopTimer(time_GPU));
	 break;
	 
     case AVERAGE_FILTER:
     printf("Average Filter \n");
	 cutilCheckError(cutStartTimer(time_GPU));
     AverageFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
	 cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	 cutilCheckError(cutStopTimer(time_GPU));
     break;
	 
     case HIGH_BOOST_FILTER:
     printf("Boost Filter \n");
	 cutilCheckError(cutStartTimer(time_GPU));
     HighBoostFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
	 cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	 cutilCheckError(cutStopTimer(time_GPU));
     break;
	 
    }
   
}



// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
            BMPInFile = argv[i+1];
	    i = i + 1;
        }
        if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
            BMPOutFile = argv[i+1];
	    i = i + 1;
        }
        if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-filter") == 0) {
            Filter = argv[i+1];
	    i = i + 1;
            if (strcmp(Filter, "sobel") == 0)
		FilterMode = SOBEL_FILTER;
			else if (strcmp(Filter, "sobel5") == 0)
		FilterMode = SOBEL_FILTER5;	
            else if (strcmp(Filter, "average") == 0)
		FilterMode = AVERAGE_FILTER;
            else if (strcmp(Filter, "boost") == 0)
		FilterMode = HIGH_BOOST_FILTER;
	 
        }
    }
}



