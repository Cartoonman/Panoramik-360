"""
Copyright (c) Steven P. Goldsmith. All rights reserved.

Created by Steven P. Goldsmith on February 4, 2017
sgoldsmith@codeferm.com
"""

"""Motion detector uses threading where possible to keep consistent FPS.

Resizes frame, sampling and uses moving average to determine change percent. Inner
rectangles are filtered out as well. This can result in better performance and
a more stable ROI.

Optional pedestrian detector using sampling, resize and motion ROI. Histogram of Oriented
Gradients ([Dalal2005]) object detector is used. You can get up to 1200%
performance boost using this method.

Optional Haar Feature-based Cascade Classifier for Object Detection. The object detector
was initially proposed by Paul Viola and improved by Rainer Lienhart.

A frame buffer is used to record 1 second before motion threshold is triggered.

sys.argv[1] = configuration file name or will default to "motiondetect.ini" if no args passed.

@author: sgoldsmith

"""

import logging, sys, os, time, datetime, threading, numpy, cv2, mjpegclient, motiondet, redis, boto3
from glob import glob
import numpy as np
import shutil
import errno

frameOk = True

UPLOAD = False
LOCAL = False

def upload_result(path, filename):
    if UPLOAD is False:
        return
    print("Uploading")
    s3 = boto3.client('s3')
    s3.upload_file(path, os.environ.get("S3_BUCKET"), '360_stream/data/' + filename)  




def markRectSize(target, rects, widthMul, heightMul, boxColor, boxThickness):
    """Mark rectangles in image"""
    for x, y, w, h in rects:
        # Calculate full size
        x2 = x * widthMul
        y2 = y * heightMul
        w2 = w * widthMul
        h2 = h * heightMul
        # Mark target
        cv2.rectangle(target, (x2, y2), (x2 + w2, y2 + h2), boxColor, boxThickness)
        label = "%dx%d" % (w2, h2)
        # Figure out text size
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]
        # Deal with possible text outside of image bounds
        if x2 < 0:
            x2 = 0
        if y2 < size[1]:
            y2 = size[1] + 2
        else:
            y2 = y2 - 2
        # Show width and height of full size image
        cv2.putText(target, label, (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        
def filterByWeight(foundLocsList, foundWghtsList, minWeight):
    """Filter out found locations by weight"""
    filteredFoundLocations = []
    filteredFoundWeights = []
    for foundWeights, foundLocations in zip(foundWghtsList, foundLocsList):
        filteredLocations = []
        filteredWeights = []
        i = 0
        # Filter by weight
        for w in foundWeights:
            if w >= minWeight:
                filteredLocations.append(foundLocations[i])
                filteredWeights.append(w)
            i += 1
        if len(filteredLocations) > 0:
            filteredFoundLocations.append(filteredLocations)
            filteredFoundWeights.append(filteredWeights)
    return filteredFoundLocations, filteredFoundWeights

def markRectWeight(target, locList, foundLocsList, foundWghtsList, widthMul, heightMul, boxColor, boxThickness):
    """Mark ROI rectangles with weight in image"""
    for location, foundLocations, foundWeights in zip(locList, foundLocsList, foundWghtsList):
        i = 0
        # Mark target
        for x, y, w, h in foundLocations:
            # Calculate full size
            x2 = x * widthMul
            y2 = y * heightMul
            w2 = w * widthMul
            h2 = h * heightMul            
            x3, y3, w3, h3 = location
            # Calculate full size
            x4 = x3 * widthMul
            y4 = y3 * heightMul
            # Mark target
            cv2.rectangle(target, (x2 + x4, y2 + y4), (x2 + x4 + w2, y2 + y4 + h2), boxColor, boxThickness)
            label = "%1.2f" % foundWeights[i]
            # Figure out text size
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]            
            # Print weight
            cv2.putText(target, label, (x2 + x4, y2 + y4 + h2 - size[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            i += 1

def markRoi(target, locList, foundLocsList, widthMul, heightMul, boxColor, boxThickness):
    """Mark ROI objects in image"""
    for location, foundLocations in zip(locList, foundLocsList):
        # Mark target
        for x, y, w, h in foundLocations:
            # Calculate full size
            x2 = x * widthMul
            y2 = y * heightMul
            w2 = w * widthMul
            h2 = h * heightMul            
            x3, y3, w3, h3 = location
            # Calculate full size
            x4 = x3 * widthMul
            y4 = y3 * heightMul
            # Mark target
            cv2.rectangle(target, (x2 + x4, y2 + y4), (x2 + x4 + w2, y2 + y4 + h2), boxColor, boxThickness)
            label = "%dx%d" % (w2, h2)
            # Figure out text size
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]            
            # Deal with possible text outside of image bounds
            if x2 < 0:
                x2 = 0
            if y2 < size[1]:
                y2 = size[1] + 2
            else:
                y2 = y2 - 2
            # Show width and height of full size image
            cv2.putText(target, label, (x2 + x4, y2 + y4), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

def saveFrame(frame, saveDir, saveFileName):
    """Save frame"""
    # Create dir if it doesn"t exist
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    cv2.imwrite("%s/%s" % (saveDir, saveFileName), frame)
    upload_result("%s/%s" % (saveDir, saveFileName), saveFileName)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise

        
def initMjpegVideo(url, socketTimeout):
    """Initialize MJPEG stream"""
    # Open MJPEG stream
    # socketFile, streamSock, boundary = mjpegclient.open(url, socketTimeout)
    # Determine image dimensions
    image = mjpegclient.getFrame()
    frameHeight, frameWidth, channels = image.shape
    return frameWidth, frameHeight

def initVidCapVideo(url):
    """Initialize VideoCapture stream"""
    # Process file or other URL with VideoCapture
    videoCapture = cv2.VideoCapture(url)
    frameHeight = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    return frameWidth, frameHeight, videoCapture, fps

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


def readVidCapFrames(logger, frameBuf, frameBufMax, videoCapture):
    """Read frames and append to buffer"""
    global frameOk
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    fpsTime = 1 / fps
    while(frameOk):
        start = time.time()
        now = datetime.datetime.now()
        frameOk, image = videoCapture.read()
        if frameOk:
            # Make sure we do not run out of memory
            if len(frameBuf) > frameBufMax:
                logger.error("Frame buffer exceeded: %d" % frameBufMax)
                frameOk = False
            else:
                # Add new image to end of list
                frameBuf.append((image, now))
            curTime = time.time()
            elapsed = curTime - start
            # Try to keep FPS for files consistent otherwise frameBufMax will be reached
            if elapsed < fpsTime:
                time.sleep(fpsTime - elapsed)
    logger.info("Exiting video stream thread")  


def config(parser):
    """Configure from INI file"""
    # Set camera related data attributes
    config.cameraName = parser.get("camera", "name")    
    config.url = parser.get("camera", "url")
    config.socketTimeout = parser.getint("camera", "socketTimeout")
    config.resizeWidthDiv = parser.getint("camera", "resizeWidthDiv")
    config.fpsInterval = parser.getfloat("camera", "fpsInterval")
    config.fps = parser.getint("camera", "fps")
    config.frameBufMax = parser.getint("camera", "frameBufMax")
    config.fourcc = parser.get("camera", "fourcc")
    config.recordFileExt = parser.get("camera", "recordFileExt")
    config.recordDir = parser.get("camera", "recordDir")
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
    global UPLOAD
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
    
    # See if we have MJPEG stream
    mjpeg = None
    if PY3:
        mjpeg = urllib.parse.urlparse(config.url).scheme == "http"
    else:
        mjpeg = urlparse.urlparse(config.url).scheme == "http"
    
    # Initialize video
    if mjpeg:
        print ("Waiting for initial frame...")
        frameWidth, frameHeight = initMjpegVideo(config.url, config.socketTimeout)
        fps = config.fps
        print ("Initial frame recieved. Synchronizing...")
    else:
        frameWidth, frameHeight, videoCapture, fps = initVidCapVideo(config.url)
        
    logger.info("OpenCV %s" % cv2.__version__)
    logger.info("URL: %s, fps: %d" % (config.url, fps))
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
        # Analyze only ~3 FPS which works well with this type of detection
        frameToCheck = int(fps / 4)
        # 0 means check every frame
        if frameToCheck < 1:
            frameToCheck = 0
        skipCount = 0         
        # Frame buffer
        frameBuf = []
        # History buffer to capture just before motion
        historyBuf = []
        recording = False
        global frameOk
        elapsedFrames = 0
        frameTotal = 0
        fileDir = None
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
        if mjpeg:
            thread = threading.Thread(target=readMjpegFrames, 
                args=(logger, 
                    frameBuf, 
                    config.frameBufMax, 
                    )
                )
        else:
            thread = threading.Thread(target=readVidCapFrames, 
                args=(logger, 
                    frameBuf, 
                    config.frameBufMax, 
                    videoCapture,)
                )
                
        thread.start()
        
        
        # Wait until buffer is full
        print ("Waiting to fill buffer.. Preloading")
        while((len(frameBuf) < fps) and (r.get('det_status').decode('utf-8') == 'GO')):
            # 1/4 of FPS sleep
            time.sleep(1)     
        print ("Buffer Filled. Beginning motion detection routine. \nContinue sending images. Use stop.py to signal stop")         
        start = time.time()
        appstart = start
        
        #mkdir_p('/tmp/raw_frames') 
        mkdir_p('/tmp/frames') 
        indx = 0
        # Loop as long as there are frames in the buffer
        while(len(frameBuf) > 0):
            # Wait until frame buffer is full
            while((len(frameBuf)) < fps and (r.get('det_status').decode('utf-8') == 'GO')):
                # 1/4 of FPS sleep
                time.sleep(1)
            
            if r.get('det_status').decode('utf-8') == 'STOP':
                print("Stop signal recieved. Clearing up buffer")
            
            # Get oldest frame
            frame = frameBuf[0][0]
            # Used for timestamp in frame buffer and filename
            now = frameBuf[0][1]
            
            
            # Buffer oldest frame
            historyBuf.append(frameBuf[0])
            # Toss oldest history frame
            if len(historyBuf) > fps:
                historyBuf.pop(0)
            # Toss oldest frame
            frameBuf.pop(0)
            frameTotal += 1
            # Calc FPS    
            elapsedFrames += 1
            curTime = time.time()
            elapse = curTime - start
            # Log FPS
            if elapse >= config.fpsInterval:
                start = curTime
                logger.debug("%3.1f FPS, frame buffer size: %d" % (elapsedFrames / elapse, len(frameBuf)))
                elapsedFrames = 0

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
                                                                                              config.erodeAmount)

            
            
            print("Processing done.")
            if config.historyImage:
                # Update history image
                historyImg = numpy.bitwise_or(bwImg, historyImg)                    
            # Threshold to trigger motion
            
            print("motion percent: " + str(motionPercent))
            
            print(len(frameBuf))
            print(len(historyBuf))

            #if motionPercent >= config.maxChange:
            #    skipCount = config.skipFrames
            #    logger.debug("Maximum motion change: %4.2f" % motionPercent)
            if not recording:

                # Construct directory name from camera name, recordDir and date
                dateStr = now.strftime("%Y-%m-%d")
                fileDir = "%s/%s/%s" % (os.path.expanduser(config.recordDir), config.cameraName, dateStr)
                # Create dir if it doesn"t exist
                if not os.path.exists(fileDir):
                    os.makedirs(fileDir)
                recording = True
            if config.mark:
                # Draw rectangle around found objects
                markRectSize(frame, movementLocations, widthMultiplier, heightMultiplier, (0, 255, 0), 2)
            if config.saveFrames:

                cv2.imwrite('/tmp/raw_frames/frame' + str(indx) + '.jpg', frame)

                indx = indx + 1
                    
            print("Frame processed")        
        
        
        
        elapsed = time.time() - appstart
        logger.info("Calculated %4.1f FPS, elapsed time: %4.2f seconds, frame total: %s" % (frameTotal / elapsed, elapsed, frameTotal))
        # Exit video streaming thread
        frameOk = False
        # Clean up
        if mjpeg:
            pass
        else:
            del videoCapture
        if config.historyImage and fileDir is not None:
            # Save history image ready for ignore mask editing
            fileName = "%s.%s" % (now.strftime("%H-%M-%S"), 'png')
            logger.info("%s/%s.png" % (fileDir, fileName))
            cv2.imwrite("%s/%s.png" % (fileDir, fileName), cv2.bitwise_not(historyImg))
            upload_result("%s/%s.png" % (fileDir, fileName), fileName)
                
                
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
    global UPLOAD, LOCAL
    if python_version == 3:
        print("is python 3")
        import configparser
        import urllib.parse
    else:
        print("not python 3")
        import ConfigParser
        import urlparse
        PY3 = False
    if len(sys.argv) > 1:
        for x in sys.argv[1:]:
            if x == 'upload':
                UPLOAD = True
            if x == 'local':
                LOCAL = True

    main(PY3)
