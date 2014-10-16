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

BMAT.cu

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
1. 

********************************************************************************************************/

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

/******************************************************************* Global Defines *************************************************************************/

#define RGB_COMPONENT_COLOR 255
#define BLOCKS 4
#define THREADS_PER_BLOCK 4

/******************************************************************* Global Variables *************************************************************************/

char *PPMInFileR = "r2.ppm";
char *PPMInFileL = "l2.ppm";
char *PPMOutFile = "output.ppm";
char PPMOutFile1[50] = "CPU_output.bmp";
char PPMOutFile2[50] = "GPU_output.bmp";

//Timer variables
unsigned int time_mem = 0;
unsigned int time_total = 0;
unsigned int time_GPU = 0;
unsigned int time_CPU = 0;

unsigned char rightImage[288][384],leftImage[288][384],rightImage1[288][384][3],leftImage1[288][384][3],dispMap[288][384],rightImagex[288*384], leftImagex[288*384], *dispMapx;


/**************************************************************** PPM Image data structures ******************************************************************/

typedef struct
{
     unsigned char red,green,blue;
} PPMPixel;

typedef struct
{
   int x, y;
   PPMPixel *data;
} PPMImage;

/******************************************************************* Function Prototypes *********************************************************************/

void bitsum(void);
void bitxor(void);
static PPMImage *readPPM(const char *filename);
void writePPM(const char *filename, PPMImage *img);
void ParseArguments(int argc, char** argv);

/********************************************************************* Device Memory **************************************************************************/

unsigned char *d_rightImage;
unsigned char *d_leftImage;
unsigned char *d_disp;

/*********************************************************************** GPU Kernel ***************************************************************************/

__global__ void DepthMAP(unsigned char *rightImage, unsigned char *leftImage, unsigned char *dispMap) {

	unsigned int ix = blockIdx.x;
	unsigned int jx = threadIdx.x;

	unsigned char i = 0, i1, x;
	unsigned int m,n,c=0,cs;
	n = ix;	
	m = jx;
	if(m < 379){
		dispMap[n*384+m] = 0;
		x = 0;																			//Initialize the threshold comparator
		i = 1;	
		while(dispMap[n*384+m] != i && x<100)											//While disparity is not computed yet and threshold is within a limit 
		{
			i = 0;																		//Initialize the distance indicator
			
			while((i < 15) && (m+i) < 379)										        //Limit the search locality to 15 pixels and avoid buffer overrun
			{
				i++;
				
				for(i1=0;i1<5;i1++)
				{
					c = c + abs(leftImage[(n)*384+m+i1] - rightImage[(n)*384+m+i+i1]);  //Formulate sum of arithmetic differences between the image features
				}
				cs = c;
				c = 0;
				if(cs < x)																//If the difference is pixel values in the set is less than the threshold, a match is obtained 
				{
					dispMap[n*384+m] = i*i;												//Save the distance value in the disparity map
					break;
				}
				
			}
			
			if(dispMap[n*384+m] !=0)													//Once the disparity value is stored, exit from the loop.
			{
				break;
			}
			
			x++;
		}
		
	}
			
}

int main(int argc, char** argv)
{	
	// Image Read to Im1 and Im2(BMP)
	ParseArguments(argc, argv);
	unsigned int i,j,k=0;
	PPMImage *image1,*image2;
    image1 = readPPM(PPMInFileR);
    image2 = readPPM(PPMInFileL);
	
	sprintf(PPMOutFile1, "CPU_%s", PPMInFileL);
	sprintf(PPMOutFile2, "GPU_%s", PPMInFileL);

	// Create Timers
	cutilCheckError(cutCreateTimer(&time_GPU));
    cutilCheckError(cutCreateTimer(&time_CPU));
	cutilCheckError(cutCreateTimer(&time_mem));
    cutilCheckError(cutCreateTimer(&time_total));
	dispMapx = (unsigned char *)malloc(288*384*sizeof(unsigned char));
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
            rightImage[i][j] = 0.2125*rightImage1[i][j][0]+0.7154*rightImage1[i][j][1]+0.0721*rightImage1[i][j][2];
            leftImage[i][j] = 0.2125*leftImage1[i][j][0]+0.7154*leftImage1[i][j][1]+0.0721*leftImage1[i][j][2];
			rightImagex[k] = rightImage[i][j];
			leftImagex[k] = leftImage[i][j];
			k++;
        }
    }
	printf("Exiting the conversion loop\n");
	
	// Start the CPU timer
	cutilCheckError(cutStartTimer(time_CPU));
	
	unsigned char i1,x;
	unsigned int m,n,c=0,cs;

	i = 1;
	
/********************************************************************* Sequential version of Algorithm *********************************************************/
	
	for(n = 0;n < 288;n++)
	{
		
		for(m = 0;m<379;m++)
		{
			x = 0;																		//Initialize the threshold comparator
			
			while(dispMapx[n*384+m] != i && x<100)										//While disparity is not computed yet and threshold is within a limit
			{
				i = 0;																	//Initialize the distance indicator
				
				while((i < 15) && (m+i) < 379)											//Limit the search locality to 15 pixels and avoid buffer overrun
				{
					i++;
					
					for(i1=0;i1<5;i1++)
					{
						c = c + abs(leftImagex[(n)*384+m+i1] - rightImagex[(n)*384+m+i+i1]);	//Formulate sum of arithmetic differences between the image features
					}
					cs = c;
					c = 0;
					if(cs < x)															//If the difference is pixel values in the set is less than the threshold, a match is obtained 
					{
						dispMapx[n*384+m] = i*i;										//Save the distance value in the disparity map
						break;
					}
					
				}
				
				if(dispMapx[n*384+m] !=0)												//Once the disparity value is stored, exit from the loop.
				break;
				
				x++;
			}
		}
	}
	// Stop the CPU timer
	cutilCheckError(cutStopTimer(time_CPU));
	
	printf("Formed the depth map\n");
	
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
	
/***************************************************************************** Parallel version of Algorithm **********************************************/

	cutilSafeCall( cudaMalloc( (void **)&d_rightImage, 288*384*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_leftImage, 288*384*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_disp, 288*384*sizeof(unsigned char)) );

	printf("malloc 1 done\n");
	
	// Star the Memory timer
	cutilCheckError(cutStartTimer(time_mem));
		
	cudaMemcpy(d_rightImage, rightImagex, 288*384*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_leftImage, leftImagex, 288*384*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	// Stop the memory timer
	cutilCheckError(cutStopTimer(time_mem));
			
	
	printf("Before launching the kernel\n");
	
	
/******************************************************************************* Call the GPU Kernel **********************************************************/
	// Start the GPU timer
	cutilCheckError(cutStartTimer(time_GPU));

	DepthMAP<<<288,384>>>(d_rightImage, d_leftImage, d_disp);
	cutilCheckMsg("kernel launch failure");
	cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel

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
	printf("After copying data to image3\n");
	image1->data = image1->data - 288*384;
	writePPM(PPMOutFile2,image1);
	printf("After writing to gpu file\n");
	printf("Timing Analysis:\n");
	printf("CPU Time: %f\n", cutGetTimerValue(time_CPU));
	printf("GPU Time: %f\n", cutGetTimerValue(time_GPU));
	printf("Memory transfer time for the GPU: %f\n", cutGetTimerValue(time_mem));
	printf("Speedup: %f\n", (cutGetTimerValue(time_CPU)/cutGetTimerValue(time_GPU)));
	printf("Percent improvement: %f%\n", (cutGetTimerValue(time_CPU)-cutGetTimerValue(time_GPU))/cutGetTimerValue(time_CPU)*100);
	return 0;

}

/***************************************************************** Read PPM Image File to Buffer *************************************************************/

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

/******************************************************************* Write PPM Image to file ********************************************************************/

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
