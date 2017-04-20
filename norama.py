from __future__ import division
"""./norama-direct -P -i /home/eclippse/Desktop/stream.jpg -o /home/eclippse/Desktop/output.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 90 -e 30 -r 30 -u 90

-a 90 -e 0 -r 0 -u 0    

u= field of view = 90 for now
a= horizontal angle = 0=outer sides, 180 is middle of image)
e= elevation (0=straight, 90=ceiling, -90=right beneath the camera)butt
r= rotation of image(leave as 0 degrees)"""

#source ~/.virtualenvs/cv/bin/activate

import subprocess
import os
from subprocess import call     #call(["ls", "-a"])

os.chdir("/usr/lib")            #change directory to where the norama directory program
"""
#PART 1 horizontal angle in depth
a= ' 180 '
os.system("./norama-direct -P -i /home/eclippse/Desktop/stream.jpg -o /home/eclippse/Desktop/output180.jpg -X 1000 -Y 1000 -k 500 -l 500 -a"+ a +"-e 0 -r 0 -u 90")


#horizontal angle of a=0 is far edges combined################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output0.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 0 -e 0 -r 0 -u 90")

#horizontal angle of a=45 is between 0 and 90 ################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output45.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 45 -e 0 -r 0 -u 90")

#horizontal angle of a=90 is left of middle#######################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output90.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 90 -e 0 -r 0 -u 90")  

#horizontal angle of a=135 is between 90 and 180 #################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output135.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 135 -e 0 -r 0 -u 90") 

#horizontal angle of a=180 is exact middle########################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output180.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 180 -e 0 -r 0 -u 90")  

#horizontal angle of a=225 is between 180 and 270 #################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output225.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 225 -e 0 -r 0 -u 90") 

#horizontal angle of a=270 is right of middle######################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output270.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 270 -e 0 -r 0 -u 90")

#horizontal angle of a=315 is between 270 and 0 #################################
os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/output315.jpg -X 1000 -Y 1000 -k 500 -l 500 -a 315 -e 0 -r 0 -u 90") """


#PART 2 pixel manipulation
#total is 2688 pixels from left to right


def mapORSLIDERIFYOUMUST( value):
    input = ((360/2688)*value) 

    p = str(input)
    print input
   

    os.system("./norama-direct -P -i /home/eclippse/Desktop/class.png -o /home/eclippse/Desktop/outputALOHA.png -X 1000 -Y 1000 -k 500 -l 500 -a "+ p +" -e 0 -r 0 -u 45") 


if __name__ == '__main__': 
    mapORSLIDERIFYOUMUST(2531) 










