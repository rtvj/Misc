#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>



typedef double FLOAT;
typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

UINT8 header[22];
UINT8 buff[230400];
//UINT8 G[110592];
//UINT8 B[110592];
UINT8 pgm[76800];

void main(int argc, char *args[]) {
    
    int fdin, fdout, bytesRead=0, bytesLeft, i, j, k=0;
    FLOAT gray_val;
    char ppm_hdr[]="P5\n#test\n320 240\n255\n";
    if(argc < 2)
    {
       printf("Usage: sharpen file.pgm\n");
       exit(-1);
    }
    else
    {


        if((fdin = open(args[1], O_RDONLY, 0644)) < 0)
        {
            printf("Error opening %s\n", args[1]);
        }
        
	char outfile[50];					// file name of the output file
	sprintf(outfile, "%s_out.ppm", args[1]);
	
        if((fdout = open(outfile, (O_RDWR | O_CREAT), 0666)) < 0)
        {
            printf("Error opening %s\n", args[1]);
        }
    }

    bytesLeft=21;

    //printf("Reading header\n");
    memset(pgm, 0, sizeof(pgm));
    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;
    } while(bytesLeft > 0);

    header[21]='\0';

    //printf("header = %s\n", header); 

    // Read RGB data
    for(i=0; i<230400; i++)
    {
        //read(fdin, (void *)&R[i], 1);
        //read(fdin, (void *)&G[i], 1);
        read(fdin, (void *)&buff[i], 1);
	
	//gray_val = (0.299*R[i]+0.587*G[i]+0.114*B[i]);
	
	//if(gray_val<0.0) gray_val=0.0;
	//if(gray_val>255.0) gray_val=255.0;
	
	//printf("gray_val: %f\n", gray_val);
	
	//pgm[i] = (UINT8)gray_val;
	//printf("pgm[%d]: %d\n", i, pgm[i]);
    }
    for(i=0;i<76800;i++){
      
      gray_val=(0.2989*buff[0+k] + 0.5870*buff[1+k] + 0.1140*buff[2+k]);
      
      pgm[i] = (UINT8)gray_val;
      k += 3;
      
    }
    write(fdout, (void *)ppm_hdr, 21);
    for(i=0; i<76800; i++)
    {
        write(fdout, (void *)&pgm[i], 1);
    }


    close(fdin);
    close(fdout);
}