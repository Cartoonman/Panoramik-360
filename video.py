import cv2
import shutil
import time
import redis
import boto3
import errno
import os
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise


def upload_result(path, filename):
    print("Uploading")
    s3 = boto3.client('s3')
    s3.upload_file(path, os.environ.get("S3_BUCKET"), '360_stream/data/' + filename)  


def get_frames(r):
    s3 = boto3.client('s3')
    s3.download_file(os.environ.get("S3_BUCKET"), '360_stream/data/frame.zip', '/tmp/frame.zip')
    
    r.set('v_frames_ready', False)
    
    mkdir_p('/tmp/frame')
    
    shutil.unpack_archive('/tmp/frame.zip', '/tmp/frame', 'zip')

    
    
    
if __name__ == '__main__':
    r = redis.from_url(os.environ.get("REDIS_URL"))
    if r.get('v_frames_ready') == 'False':
        time.sleep(10)
    get_frames(r)
    
    image = cv2.imread('/tmp/frame/img' + str(0) + '.jpg')
    frameResizeHeight = image.shape[0]
    frameResizeWidth = image.shape[1]
    
    videoWriter = cv2.VideoWriter("%s/%s" % ('/tmp', 'video.avi'), 
                          cv2.VideoWriter_fourcc('X','V','I','D'), 
                          2, 
                          (frameResizeWidth*2, frameResizeHeight*3), 
                          True
                          )
    
                
    for x in range(int(r.get('v_frames_count'))):
    
        image1 = cv2.imread('/tmp/frame/img' + str(x) + '.jpg')
        image2 = cv2.imread('/tmp/frame/imgw' + str(x) + '.jpg')
        image3 = cv2.imread('/tmp/frame/imgma' + str(x) + '.jpg')
        image4 = cv2.imread('/tmp/frame/imgd' + str(x) + '.jpg')
        image5 = cv2.imread('/tmp/frame/imgmaG' + str(x) + '.jpg')
        image6 = cv2.imread('/tmp/frame/imgmaBW' + str(x) + '.jpg')
        video_frame = np.zeros((image1.shape[0]*3,image1.shape[1]*2,3), np.uint8)
        
        video_frame[0:image1.shape[0], 0:image1.shape[1]] = image1
        video_frame[image1.shape[0]:image1.shape[0]*2, 0:image1.shape[1]] = image2
        video_frame[image1.shape[0]*2:image1.shape[0]*3, 0:image1.shape[1]] = image3
        video_frame[0:image1.shape[0], image1.shape[1]:image1.shape[1]*2] = image4
        video_frame[image1.shape[0]:image1.shape[0]*2, image1.shape[1]:image1.shape[1]*2] = image5
        video_frame[image1.shape[0]*2:image1.shape[0]*3, image1.shape[1]:image1.shape[1]*2] = image6
        videoWriter.write(video_frame)
    
    videoWriter.release()
    del videoWriter
    upload_result('/tmp/video.avi', 'video.avi')
    
    
    
