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

#define CLAMP_8bit(x) max(0, min(255, (x)))
#define RGB_COMPONENT_COLOR 255


typedef struct
{
     unsigned char red,green,blue;
} PPMPixel;

typedef struct
{
     int x, y;
     PPMPixel *data;
} PPMImage;

unsigned int time_CPU = 0;

// Includes: local

enum {SOBEL_FILTER=1, HIGH_BOOST_FILTER};


char *Filter = "sobel";
int FilterMode  = SOBEL_FILTER;

// Functions
static PPMImage *readPPM(const char *filename);
void writePPM(const char* filename, PPMImage* img);
void ParseArguments(int argc, char** argv);


const int FILTER_RADIUS = 1;
//const int FILTER_RADIUS = 2; // for sobel 5x5

const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA   = FILTER_DIAMETER * FILTER_DIAMETER;

const int EDGE_VALUE_THRESHOLD = 70;
//const int EDGE_VALUE_THRESHOLD = 500; //sobel 5
const int HIGH_BOOST_FACTOR = 10;

#include "filter_kernel.cu"


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
    PPMImage *input_image=NULL, *output_image=NULL;
    output_image = (PPMImage *)malloc(sizeof(PPMImage));
    input_image = readPPM(PPMInFileL);
    printf("Running %s filter\n", Filter);
    out = (unsigned char *)malloc();

    printf("Computing the CPU output\n");
    printf("Image details: %d by %d = %d , imagesize = %d\n", input_image->x, input_image->y, input_image->x * input_image->y, input_image->x * input_image->y);
    
	cutilCheckError(cutStartTimer(time_CPU));
	if(FilterMode == SOBEL_FILTER){
	printf("Running Sobel\n");
	CPU_Sobel(intput_image->data, output_image, input_image->x, input_image->y);
	}
	else if(FilterMode == HIGH_BOOST_FILTER){
	printf("Running boost\n");
	CPU_Boost(data, out, dib.width, dib.height);
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