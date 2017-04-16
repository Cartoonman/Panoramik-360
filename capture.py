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

    while(True):
        subprocess.call(["gphoto2","--capture-image-and-download", "--filename=stream.jpg", "--force-overwrite"])




        print ("Uploading")
        upload_result(r,  "stream.jpg")
        print(r.get('det_ready').decode('utf-8'))
        while r.get('det_ready').decode('utf-8') == "True":
            time.sleep(5)
        time.sleep(20)

if __name__ == '__main__':
    takePicture()
