"""
Copyright (c) Steven P. Goldsmith. All rights reserved.

Created by Steven P. Goldsmith on February 3, 2017
sgoldsmith@codeferm.com
"""

"""Socket based MJPEG frame grabber.

@author: sgoldsmith

"""

import cv2
import os
import boto3
import redis
import time
from glob import glob

    
def get_image(r):
    s3 = boto3.client('s3')
    s3.download_file(os.environ.get("S3_BUCKET"), '360_stream/stream.jpg', 'stream.jpg')
    
    r.set('det_ready', False)
    
    return cv2.imread('stream.jpg')    
    
    

def getFrame():
    """Get raw frame data from stream and decode"""
    r = redis.from_url(os.environ.get("REDIS_URL"))

    while r.get('det_ready').decode('utf-8') == "False":
        time.sleep(1)
        if (r.get('det_status').decode('utf-8') == 'STOP'):
            return None
    print("Getting Image")
    frame = get_image(r)
    
    #jpeg = socketFile.read(getFrameLength(socketFile, boundary))
    #return jpeg, cv2.imdecode(numpy.fromstring(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    
    
    return frame
    
