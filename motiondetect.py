import logging
import os
import time, datetime
import numpy
import threading
import cv2
import mjpegclient, motiondet
import redis 
import boto3
from glob import glob
import shutil
import errno

frameOk = True
UPLOAD = False
LOCAL = False

def upload_result(path, filename):
    global UPLOAD
    if UPLOAD is False:
        return
    print("Uploading")
    s3 = boto3.client('s3')
    s3.upload_file(path, os.environ.get("S3_BUCKET"), '360_stream/data/' + filename)  


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise

        
def initMjpegVideo():
    """Initialize MJPEG stream"""
    # Open MJPEG stream
    # Determine image dimensions
    image = mjpegclient.getFrame()
    frameHeight, frameWidth, channels = image.shape
    return frameWidth, frameHeight



def readMjpegFrames(logger, frameBuf, frameBufMax):
    """Read frames and append to buffer"""
    global frameOk, LOCAL
    if LOCAL:
        res = (cv2.imread(x) for x in sorted(glob('photos/*.JPG')))
    while(frameOk):
        now = datetime.datetime.now()
        # Make sure thread doesn't hang in case of socket time out, etc.
        image = None
        if LOCAL:
            try:
                image = next(res)
            except StopIteration:
                break
        
        else:
            image = mjpegclient.getFrame()
        #frameOk = len(jpeg) > 0
        if image is not None:
            # Make sure we do not run out of memory
            if len(frameBuf) > frameBufMax:
                logger.error("Frame buffer exceeded: %d" % frameBufMax)
                frameOk = False
            else:
                # Add new image to end of list
                frameBuf.append((image, now))
    logger.info("Exiting video stream thread")  




def config(parser):
    """Configure from INI file"""
    # Set camera related data attributes
    config.cameraName = parser.get("camera", "name")    
    config.socketTimeout = parser.getint("camera", "socketTimeout")
    config.resizeWidthDiv = parser.getint("camera", "resizeWidthDiv")
    config.frameBufMax = parser.getint("camera", "frameBufMax")
    config.fourcc = parser.get("camera", "fourcc")
    config.detectType = parser.get("camera", "detectType")
    config.mark = parser.getboolean("camera", "mark")
    config.saveFrames = parser.getboolean("camera", "saveFrames")
    # Set motion related data attributes
    config.ignoreMask = parser.get("motion", "ignoreMask")
    config.kSize = eval(parser.get("motion", "kSize"), {}, {})
    config.alpha = parser.getfloat("motion", "alpha")
    config.blackThreshold = parser.getint("motion", "blackThreshold")
    config.maxChange = parser.getfloat("motion", "maxChange")
    config.skipFrames = parser.getint("motion", "skipFrames")
    config.startThreshold = parser.getfloat("motion", "startThreshold")
    config.stopThreshold = parser.getfloat("motion", "stopThreshold")
    config.historyImage = parser.getboolean("motion", "historyImage")
    # Set contour related data attributes
    config.dilateAmount = parser.getint("motion", "dilateAmount")
    config.erodeAmount = parser.getint("motion", "erodeAmount")
    # Set pedestrian detect related data attributes
    config.hitThreshold = parser.getfloat("pedestrian", "hitThreshold")
    config.winStride = eval(parser.get("pedestrian", "winStride"), {}, {})
    config.padding = eval(parser.get("pedestrian", "padding"), {}, {})
    config.scale0 = parser.getfloat("pedestrian", "scale0")
    config.minWeight = parser.getfloat("pedestrian", "minWeight")
    # Set cascade related data attributes
    config.cascadeFile = parser.get("cascade", "cascadeFile")
    config.scaleFactor = parser.getfloat("cascade", "scaleFactor")
    config.minNeighbors = parser.getint("cascade", "minNeighbors")
    config.minWidth = parser.getint("cascade", "minWidth")
    config.minHeight = parser.getint("cascade", "minHeight")


def main(PY3):
    """Main function"""
    configFileName = "motiondetect.ini"
    r = redis.from_url(os.environ.get("REDIS_URL"))
    r.set('det_status', 'GO')
    parser = None
    if PY3:
        parser = configparser.SafeConfigParser()
    else: 
        parser = ConfigParser.SafeConfigParser()
    # Read configuration file
    parser.read(configFileName)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(parser.get("logging", "level"))
    formatter = logging.Formatter(parser.get("logging", "formatter"))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Load values from ini file
    config(parser)

    # Initialize video
    print ("Waiting for initial frame...")
    frameWidth, frameHeight = initMjpegVideo()
    print ("Initial frame recieved. Synchronizing...")

        
    logger.info("OpenCV %s" % cv2.__version__)
    logger.info("Stream: MJPEG")
    logger.info("Resolution: %dx%d" % (frameWidth, frameHeight))
    
    # Make sure we have values > 0
    if frameWidth > 0 and frameHeight > 0:
        # Motion detection generally works best with 320 or wider images
        widthDivisor = int(frameWidth / config.resizeWidthDiv)
        if widthDivisor < 1:
            widthDivisor = 1
        frameResizeWidth = int(frameWidth / widthDivisor)
        frameResizeHeight = int(frameHeight / widthDivisor)
        logger.info("Resized to: %dx%d" % (frameResizeWidth, frameResizeHeight))
        # Used for full size image marking
        widthMultiplier = int(frameWidth / frameResizeWidth)
        heightMultiplier = int(frameHeight / frameResizeHeight)
   
        # Frame buffer
        frameBuf = []
        
        elapsedFrames = 0
        frameTotal = 0
        if config.historyImage:
            # Create black history image
            historyImg = numpy.zeros((frameResizeHeight, frameResizeWidth), numpy.uint8)
            
        # Read ignore mask image if set
        if config.ignoreMask != "":
            maskImg = cv2.imread(config.ignoreMask, 0)
            logger.info("Using ignore mask: %s" % config.ignoreMask)            
        else:
            config.ignoreMask = None
            maskImg = None   
        movingAvgImg = None
        
        # Kick off video stream thread

        thread = threading.Thread(target=readMjpegFrames, 
            args=(logger, 
                frameBuf, 
                config.frameBufMax, 
                )
            )
        thread.start()
        
        
        # Wait for base frame
        print ("Waiting for first frame")
        while((len(frameBuf) < 1) and (r.get('det_status').decode('utf-8') == 'GO')):
            time.sleep(1)     
        print ("Initial frame recieved. Beginning motion detection routine. \nContinue sending images. Use stop.py to signal stop")         
        
        #mkdir_p('/tmp/raw_frames') 
        mkdir_p('/tmp/frames') 
        indx = 0
        # Loop as long as there are frames in the buffer
        while(len(frameBuf) > 0):
            if r.get('det_status').decode('utf-8') == 'STOP':
                print("Stop signal recieved. Clearing up buffer")
            
            # Get oldest frame
            frame = frameBuf[0][0]
            print(str(frameBuf[0][1]))
            # Toss oldest frame
            frameBuf.pop(0)
            frameTotal += 1

            # Resize image if not the same size as the original
            if frameResizeWidth != frameWidth:
                resizeImg = cv2.resize(frame, (frameResizeWidth, frameResizeHeight), interpolation=cv2.INTER_NEAREST)
            else:
                resizeImg = frame
                
            print("Processing motion detection..")
            # Detect motion
            movingAvgImg, grayImg, bwImg, motionPercent, movementLocations = motiondet.detect(movingAvgImg, 
                                                                                              maskImg, 
                                                                                              resizeImg, 
                                                                                              config.kSize, 
                                                                                              config.alpha, 
                                                                                              config.blackThreshold, 
                                                                                              config.maxChange, 
                                                                                              config.dilateAmount, 
                                                                                              config.erodeAmount,
                                                                                              config.detectType,
                                                                                              config.hitThreshold, 
                                                                                              config.winStride,
                                                                                              config.padding,
                                                                                              config.scale0,
                                                                                              config.minWeight,
                                                                                              widthMultiplier,
                                                                                              heightMultiplier,
                                                                                              r
                                                                                              )

            print("Processing done.")
            if config.historyImage:
                # Update history image
                historyImg = numpy.bitwise_or(bwImg, historyImg)                    
            # Threshold to trigger motion
            
            print("motion percent: " + str(motionPercent) + " frame count:" + str(frameTotal))
            
            if config.saveFrames:

                cv2.imwrite('/tmp/raw_frames/frame' + str(indx) + '.jpg', frame)

                indx = indx + 1
                    
            print("Frame processed")    
            while((len(frameBuf)) < 1 and (r.get('det_status').decode('utf-8') == 'GO')):
                time.sleep(1)    
                
        print("Cleaning up, uploading data")
        # Exit video streaming thread
        global frameOk
        frameOk = False
        
        # Clean up
        if config.historyImage:
            # Save history image ready for ignore mask editing
            cv2.imwrite("/tmp/mask.png", cv2.bitwise_not(historyImg))
            upload_result("/tmp/mask.png", 'mask.png')
                
                
        shutil.make_archive('/tmp/frame_zip', 'zip', '/tmp/frames')
        #shutil.make_archive('/tmp/raw_frame_zip', 'zip', '/tmp/raw_frames/')
        
        upload_result('/tmp/frame_zip.zip', 'frame.zip')
        r.set('v_frames_ready', True)
        r.set('v_frames_count', motiondet.indx)
        #upload_result('/tmp/raw_frame_zip.zip', 'raw_frame.zip')
                


if __name__ == '__main__':
    import sys    
    # prints whether python is version 3 or not
    python_version = sys.version_info.major
    PY3 = True
    if python_version == 3:
        print("is python 3")
        import configparser
    else:
        print("not python 3")
        import ConfigParser
        PY3 = False
    if len(sys.argv) > 1:
        for x in sys.argv[1:]:
            if x == 'upload':
                UPLOAD = True
            if x == 'local':
                LOCAL = True

    main(PY3)
