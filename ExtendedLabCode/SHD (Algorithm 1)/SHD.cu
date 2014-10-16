/********************************************************************************************************
                                   ADVANCED COMPUTER ARCHITECTURE
                                            Spring 2014
                                         Final Project Code

*********************************************************************************************************
                                CUDA IMPLEMENTATION OF DISPARITY MAP COMPUTATION
							    USING BLOCK MATCHING & SUM OF HAMMING DISTANCES
*********************************************************************************************************

Authors: Akshay Hodigere Arunkumar
         Rutvij Girish Kharkanis

SHD.cu

Required include files: stdio.h
						stdlib.h
						math.h
						sys/types.h
						sys/stat.h
						unistd.h
						fcntl.h
						stdint.h
						errno.h
						assert.h
						string.h
						sys/io.h
						cutil_inline.h

References:
1. http://en.wikipedia.org/wiki/Hamming_distance
2. https://siddhantahuja.wordpress.com/tag/matlab-code/

Description: The program is the implementation of the Sum of Hamming Distances algorithm to find the similar features in the left and right images . 
			 It works on the principle that if two particular sections of the images of an object differ at multiple pixel locations, 
			 then the object is not found. The section matching is continued till 15 locations and the section where the hamming distance 
			 is found to be the lowest is considered to be the object of interest. Then it is determined as to 
			 how much this section has drifted in its left image as compared to the right image 
			 and this drift is directly mapped as the depth of the object.
********************************************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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

/******************************************************************* Global Defines *************************************************************************/

#define RGB_COMPONENT_COLOR 255
#define BLOCKS 4
#define THREADS_PER_BLOCK 4

/******************************************************************* Global Variables *************************************************************************/

char *PPMInFileL = "l1.ppm";
char *PPMInFileR = "r1.ppm";
char *PPMOutFile = "output.ppm";
char PPMOutFile1[50] = "CPU_output.bmp";
char PPMOutFile2[50] = "GPU_output.bmp";

//Timer variables
unsigned int time_mem = 0;
unsigned int time_total = 0;
unsigned int time_GPU = 0;
unsigned int time_CPU = 0;

unsigned char rightImage[288][384],leftImage[288][384],rightImage1[288][384][3],leftImage1[288][384][3],dispMap[288][384],rightImagex[288*384], leftImagex[288*384], *dispMapx;
unsigned char rightWindow[15],leftWindow[15],bloc3[15];
unsigned int distance;

/*************************************************************** PPM Image data structures ******************************************************************/
typedef struct
{
     unsigned char red,green,blue;
} PPMPixel;

typedef struct
{
     int x, y;
     PPMPixel *data;
} PPMImage;

/****************************************************************** Function Prototypes **************************************************************************/

void bitsum(void);
void bitxor(void);
static PPMImage *readPPM(const char *filename);
void writePPM(const char *filename, PPMImage *img);
void ParseArguments(int argc, char** argv);


/********************************************************************* Device Memory **************************************************************************/

unsigned char *d_rightImage;
unsigned char *d_leftImage;
unsigned char *d_disp;
unsigned char *d_rightWindow;
unsigned char *d_leftWindow;
unsigned char *d_bloc3;

/*********************************************************************** GPU Kernel ***************************************************************************/

__global__ void DepthMAP(unsigned char *rightImage, unsigned char *leftImage, unsigned char *dispMap) {

	unsigned int ix = blockIdx.x;
	unsigned int jx = threadIdx.x;
	
	volatile unsigned char rightWindow[15];
	volatile unsigned char leftWindow[15];
	volatile unsigned char d_bloc3[15];

	volatile unsigned char  dispMin = 0, dispMax = 15, dispRange, win = 4,i1,j1,position, k, distance, min1;
	unsigned int i,j;

	i=ix+win;
	j=jx+win;
	if(i<287 && j<364)
	{
		dispMap[i*384 + j] = 0;
		min1 = 0;
		position = 0;
		
		for(j1=0;j1<15;j1++)
		{	
				rightWindow[j1] = rightImage[i*384+(j-win +j1)];					// Take a section of the right image
		}
		__syncthreads();
		
		for(dispRange = dispMin;dispRange<dispMax;dispRange++)
		{
			if((j+win+dispRange)<384)
			{
					
				for(j1=0;j1<15;j1++)
				{
					leftWindow[j1] = leftImage[i*384 + (j-win +j1+dispRange)];		//Take a section of the left image upto a locality of 15 pixels in a single row
				}																	//to compare with the section of the right image
				
			__syncthreads();
			distance = 0;
			//bitxor();
		
				for(j1=0;j1<15;j1++)
				{
					d_bloc3[j1]=rightWindow[j1] ^ leftWindow[j1];					//Find the Hamming distance between the sections
				}
			
			__syncthreads();
			//bitsum();
			
				for(j1=0;j1<15;j1++)
				{
					for(k=0;k<8;k++)
					{
						if(d_bloc3[j1] & 1)
							distance++;
						d_bloc3[j1]=d_bloc3[j1] >> 1;
					}
				}
				

				if (dispRange == dispMin)											//Evaluate the lowest hamming distance in the above said locality and
				{																	//Save the location of the pixels of the lowest hamming distance in the 
					min1 = distance;												//Disparity Matrix
				}

				if(min1>distance)
				{
					min1 = distance;
					position = dispRange;
				}
				
			}

		}
		__syncthreads();

		dispMap[i*384 + j] = position*position;

		__syncthreads();

	
	}

	
}

int main(int argc, char ** argv)
{
	unsigned char  dispMin = 0, dispMax = 15, dispRange, win = 4,j1,position;
	unsigned int i,j,min1,k=0;
	ParseArguments(argc, argv);
    PPMImage *image1,*image2;
    image1 = readPPM(PPMInFileL);
    image2 = readPPM(PPMInFileR);
	dispMapx = (unsigned char *)malloc(288*384*sizeof(unsigned char));
	
	sprintf(PPMOutFile1, "CPU_%s", PPMInFileL);
	sprintf(PPMOutFile2, "GPU_%s", PPMInFileL);
	
	// Create Timers
	cutilCheckError(cutCreateTimer(&time_GPU));
    cutilCheckError(cutCreateTimer(&time_CPU));
	cutilCheckError(cutCreateTimer(&time_mem));
    cutilCheckError(cutCreateTimer(&time_total));	

	
/*

Convert single dimensional array into three dimensional array

*/

    for(i=0;i<288;i++)
    {
        for(j=0;j<384;j++)
        {
            rightImage1[i][j][0] = image1->data->red;
            rightImage1[i][j][1] = image1->data->green;
            rightImage1[i][j][2] = image1->data->blue;

            leftImage1[i][j][0] = image2->data->red;
            leftImage1[i][j][1] = image2->data->green;
            leftImage1[i][j][2] = image2->data->blue;

            image1->data++;
            image2->data++;
/*
Convert the Image into grayscale
*/
            rightImage[i][j] = 0.2125*rightImage1[i][j][0]+0.7154*rightImage1[i][j][1]+0.0721*rightImage1[i][j][2];
            leftImage[i][j] = 0.2125*leftImage1[i][j][0]+0.7154*leftImage1[i][j][1]+0.0721*leftImage1[i][j][2];
			rightImagex[k] = rightImage[i][j];
			leftImagex[k] = leftImage[i][j];
			k++;
        }
    }
/***************************************************************** Sequential version of Algorithm *********************************************************/
	
	// Start the CPU timer
	cutilCheckError(cutStartTimer(time_CPU));

	for(i=win;i<287-win;i++)
	{
		for(j=win;j<383-win-dispMax;j++)
		{
			
			dispMapx[i*384 + j] = 0;
			min1 = 0;
			position = 0;
			
			for(j1=0;j1<15;j1++)
			{	
					rightWindow[j1] = rightImagex[i*384+(j-win +j1)];					// Take a section of the right image
			}																
			
			for(dispRange = dispMin;dispRange<dispMax;dispRange++)
			{
				if((j+win+dispRange)<384)
				{
						
					for(j1=0;j1<15;j1++)
					{
						leftWindow[j1] = leftImagex[i*384 + (j-win +j1+dispRange)];		//Take a section of the left image upto a locality of 15 pixels in a single row
					}																	//to compare with the section of the right image

					distance = 0;
					bitxor();															//Find the Hamming distance between the sections
	
					if (dispRange == dispMin)
					{
						min1 = distance;
					}

					if(min1>distance)
					{
						min1 = distance;
						position = dispRange;
					}
					
				}

			}

			dispMapx[i*384 + j] = position*position;
		}
	}
	// Stop the CPU timer
	cutilCheckError(cutStopTimer(time_CPU));

    image2->data = image2->data - 288*384;
	for(i=0;i<288;i++)
    {
        for(j=0;j<384;j++)
        {
		unsigned char temp = 0;
		if(j<382 && i<286)
			temp = (dispMapx[i*384+j]+dispMapx[i*384+j+1]+dispMapx[i*384+j+2]+dispMapx[(i+1)*384+j]+dispMapx[(i+1)*384+j+1]+dispMapx[(i+1)*384+j+2]+dispMapx[(i+2)*384+j]+dispMapx[(i+2)*384+j+1]+dispMapx[(i+2)*384+j+2])/9;

		image2->data->red=temp;
		image2->data->green=temp;
		image2->data->blue=temp;
		image2->data++;
        }
    }
	
    image2->data = image2->data - 288*384;
    writePPM(PPMOutFile1,image2);
	printf("CPU done\n");
	
/****************************************************************** Parallel version of Algorithm ***********************************************************/

	cutilSafeCall( cudaMalloc( (void **)&d_rightImage, 288*384*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_leftImage, 288*384*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_disp, 288*384*sizeof(unsigned char)) );
	
	printf("malloc 1 done\n");

	
	cutilSafeCall( cudaMalloc( (void **)&d_rightWindow, 9*9*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_leftWindow, 9*9*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_bloc3, 9*9*sizeof(unsigned char)) );

	printf("malloc 2 done\n");
	
		
	// Star the Memory timer
	cutilCheckError(cutStartTimer(time_mem));


	cudaMemcpy(d_rightImage, rightImagex, 288*384*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_leftImage, leftImagex, 288*384*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	
	// Stop the memory timer
	cutilCheckError(cutStopTimer(time_mem));	
	
	printf("Before launching the kernel\n");
	
	// Start the GPU timer
	cutilCheckError(cutStartTimer(time_GPU));
	
/********************************************************************* Call the GPU Kernel ********************************************************************/

	DepthMAP<<<288,384>>>(d_rightImage, d_leftImage, d_disp);			// Launch the kernel with resolution of the images as the size of grid and number of threads
	cutilCheckMsg("kernel launch failure");
	cutilSafeCall( cudaThreadSynchronize() ); 							// Have host wait for kernel
	
	// Stop the GPU timer
	cutilCheckError(cutStopTimer(time_GPU));

	printf("After running kernel\n");
		
	//Start the memory timer
	cutilCheckError(cutStartTimer(time_mem));

	cudaMemcpy(dispMapx, d_disp, 288*384*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Stop the memory timer
	cutilCheckError(cutStopTimer(time_mem));

	printf("After memcpy of dev to host\n");
	image1->data = image1->data - 288*384;
	for(i=0;i<288;i++)
    {
        for(j=0;j<384;j++)
        {
            unsigned char temp = 0;
		if(j<382 && i<286)
			temp = (dispMapx[i*384+j]+dispMapx[i*384+j+1]+dispMapx[i*384+j+2]+dispMapx[(i+1)*384+j]+dispMapx[(i+1)*384+j+1]+dispMapx[(i+1)*384+j+2]+dispMapx[(i+2)*384+j]+dispMapx[(i+2)*384+j+1]+dispMapx[(i+2)*384+j+2])/9;

		image1->data->red=temp;
		image1->data->green=temp;
		image1->data->blue=temp;
		image1->data++;
        }
    }
	printf("After copying data to image1\n");
    image1->data = image1->data - 288*384;
    writePPM(PPMOutFile2,image1);
	printf("After writing to gpu file\n");
	
/************************************************* Timing Analysis of the CPU & GPU Execution of the Algorithm ********************************************************************/
	
	printf("Timing Analysis:\n");
	printf("CPU Time: %f\n", cutGetTimerValue(time_CPU));
	printf("GPU Time: %f\n", cutGetTimerValue(time_GPU));
	printf("Memory transfer time for the GPU: %f\n", cutGetTimerValue(time_mem));
	printf("Speedup: %f\n", (cutGetTimerValue(time_CPU)/cutGetTimerValue(time_GPU)));
	printf("Percent improvement: %f%\n", ((cutGetTimerValue(time_CPU)-cutGetTimerValue(time_GPU))/cutGetTimerValue(time_CPU)*100));

	return 0;
}

/********************************************************************* EX-OR the blocks ********************************************************************/

void bitxor(void)
{
	unsigned char i;

	for(i=0;i<15;i++)
	{
	
		bloc3[i]=rightWindow[i] ^ leftWindow[i];

	}
	bitsum();
}

/********************************************************************** SUM the blocks *******************************************************************/

void bitsum(void)
{
	unsigned char i,k;
	for(i=0;i<15;i++)
	{
		for(k=0;k<8;k++)
		{
		   if(bloc3[i] & 1)
				distance++;
		   bloc3[i]=bloc3[i] >> 1;
		}
	}
}

/****************************************************************** Read PPM Image File to Buffer *************************************************************/


static PPMImage *readPPM(const char *filename)
{
	 char buff[4];
	 PPMImage *img;
	 FILE *fp;
	 int c, rgb_comp_color;
	 //open PPM file for reading
	 fp = fopen(filename, "rb");
	 if (!fp) {
		  fprintf(stderr, "Unable to open file '%s'\n", filename);
		  exit(1);
	 }

	 //read image format
	 if (!fgets(buff, sizeof(buff), fp)) {
		  perror(filename);
		  exit(1);
	 }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
            printf("%d,%d",img->x,img->y);
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

/******************************************************************* Write PPM Image to file *******************************************************************/

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}
void ParseArguments(int argc, char** argv)
{
	int i;
    for (i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
            PPMInFileL = argv[i+1];
			PPMInFileR = argv[i+2];
	    i = i + 2;
        }
        if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
            PPMOutFile = argv[i+1];
	    i = i + 1;
        }
        
    }
}
