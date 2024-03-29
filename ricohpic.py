#this python program will:
#(1)take a picture 
#(2)get camera info and manipulate it inside the Python program
#(3)get list of files on camera and snip off the handle for the last image taken 
#(4)download the last image to local disk

import subprocess
import time
import re
import boto3
import os
import redis
from glob import glob


"""
Uploads the resulting processed image to S3 for debugging/presenting to users to show the
work the server did in identifying regions
"""    
def upload_result(r, filename):
    s3 = boto3.client('s3')
    s3.upload_file(filename, os.environ.get("S3_BUCKET"), '360_stream/stream.jpg', {'ACL': 'public-read'})  
    r.set('det_ready', True)
    r.set('srv_ready', True)



## example of taking a picture
def takePicture():

    r = redis.from_url(os.environ.get("REDIS_URL"))
    
    """p = subprocess.Popen(["ptpcam", "-R", "0x100e"])
    time.sleep(5)
    #p.kill()
    # example of grabbing device info and using it in your python program.
    #ptpinfo = subprocess.Popen(["ptpcam", "--info"], stdout=subprocess.PIPE)
    #time.sleep(5)

    # although this simply prints to stdout, you can parse
    # the response for your program
    #for line in ptpinfo.stdout.readlines():
    #    print(line.rstrip())



    time.sleep(1)

    
    # find the last picture taken. Modify to parse for date or other
    files = []
    listFiles = subprocess.Popen(["ptpcam", "-L"], stdout=subprocess.PIPE)
    for line in listFiles.stdout.readlines():
        files.append(line.rstrip())
    lastLine = files[len(files) - 2].decode('utf-8').split(" ")
    lastPicture = lastLine[0][:-1]

    print("The handle for the last picture taken is " + lastPicture)

    # download the picture
    ptpcommand = "--get-file=" + lastPicture

    k = subprocess.check_output(["ptpcam", ptpcommand ]).decode('utf-8')
    p = re.compile("\"(.*?)\"")
    filename = ""
    match = p.search(k)
    if match is not None:
        filename = match.group(1) 

        
        
    if filename == "":
    	return"""
    imglist = sorted(glob('photos/*.JPG'))
    	

    for x in imglist:
        print ("Uploading")
        print x
        upload_result(r,  x)
        print(r.get('det_ready').decode('utf-8'))
        #while r.get('det_ready').decode('utf-8') == "True":
        #    time.sleep(1)
              
    
if __name__ == '__main__':
    takePicture()
