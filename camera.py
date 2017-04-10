import cv2
import ricohpic
import time
import boto3
import os
import redis

def get_image():
    s3 = boto3.client('s3')
    s3.download_file(os.environ.get("S3_BUCKET"), '360_stream/stream.jpg', 'stream.jpg')
    
    r = redis.from_url(os.environ.get("REDIS_URL"))
    
    r.set('srv_ready', False)
    
    return cv2.imread('stream.jpg')





class VideoCamera(object):  
    def get_frame(self):
        #success, image = self.video.read()
        #image = cv2.imread(ricohpic.takePicture())
        r = redis.from_url(os.environ.get("REDIS_URL"))
        while r.get('srv_ready') == "False":
            time.sleep(5)
        print("Getting Image")
        image = get_image()
        
        
        
        image = cv2.resize(image,None,fx=0.10, fy=0.10, interpolation = cv2.INTER_CUBIC)
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        
        
        return jpeg.tobytes()
    
