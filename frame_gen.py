############################################################
#
# Generates a test PGM format frame in a file to test load.
#
# Writes 240 lines of 320 bytes for color frame that is
# 320 columns x 240 rows with 3 bytes per pixel
#
############################################################

import sys
import struct

raw_input('hit any key to generate test frames')

myfile = open('test_frame.ppm', 'wb')

myfile.write('P6\n')
myfile.write('#test\n')
myfile.write('720 480\n')
myfile.write('255\n')

# Write out each R, G, B pixel for 720x480
for i in range(345600):
	red = i / 1356
	grn = 255 - red
	blu = (red+grn) / 2
	outbyte = '%c' % red
	myfile.write(outbyte)
	outbyte = '%c' % grn
	myfile.write(outbyte)
	outbyte = '%c' % blu
	myfile.write(outbyte)

#myfile.write('P6\n')
#myfile.write('#test\n')
#myfile.write('320 240\n')
#myfile.write('255\n')

# Write out each R, G, B pixel for 320x240
#for i in range(76800):
#	red = i / 302
#	grn = 255 - red
#	blu = (red+grn) / 2
#	outbyte = '%c' % red
#	myfile.write(outbyte)
#	outbyte = '%c' % grn
#	myfile.write(outbyte)
#	outbyte = '%c' % blu
#	myfile.write(outbyte)


myfile.close()
