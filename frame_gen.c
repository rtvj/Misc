#include <unistd.h>
#include <stdio.h>

// Example test frame to be used with ECEN 5033 lab for digest
// computations.
//
// Just compile with gcc frame_gen.c -o framegen in Linux or cygwin
//
// Run ./framegen testfraemc.ppm
//
// Test view with IrfanView in Windows or xv in Linux
//

char outfilename[256]="test_frame.ppm";
unsigned char testFrameBuffer[3*345600];

int main(int argc, char *argv[])
{
    FILE *fptr;
    int i, frameIdx=0;
    unsigned char red, green, blue;

    if(argc < 2)
    {
	printf("Will default to output file=test_frame.ppm\n");
    }
    else
    {
        sscanf(argv[1], "%s", &outfilename);
        printf("Will output file to %s\n", outfilename);
    }

    if((fptr=fopen(outfilename, "wb")) == (FILE *)0)
    {
        printf("file open failure\n");
        exit(-1);
    }

    fprintf(fptr, "P6\n");
    fprintf(fptr, "#test\n");
    fprintf(fptr, "720 480\n");
    fprintf(fptr, "255\n");

    // Write out each R, G, B pixel for 720x480 - Has rainbow look
    // with green near top ranging down to violet/pink color at bottom.
    // This can be viewed with IrfanView in Windows or VNC in Linux.
    for(i=0;i<345600;i++)
    {
	red = i / 1356;
	green = 255 - red;
	blue = (red+green) / 2;

	// Fill out in memory frame buffer
	testFrameBuffer[frameIdx]=red; frameIdx++;
	testFrameBuffer[frameIdx]=green; frameIdx++;
	testFrameBuffer[frameIdx]=blue; frameIdx++;

	//fwrite(&red, sizeof(unsigned char), 1, fptr);
	//fwrite(&green, sizeof(unsigned char), 1, fptr);
	//fwrite(&blue, sizeof(unsigned char), 1, fptr);

        // Write out from test frame buffer	
	fwrite(&testFrameBuffer[(i*3)], sizeof(unsigned char), 1, fptr);
	fwrite(&testFrameBuffer[(i*3)+1], sizeof(unsigned char), 1, fptr);
	fwrite(&testFrameBuffer[(i*3)+2], sizeof(unsigned char), 1, fptr);

    }

    fclose(fptr);
}
