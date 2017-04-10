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

import configparser, logging, sys, os, time, datetime, threading, numpy, cv2, urllib.parse, mjpegclient, motiondet, pedestriandet, cascadedet, scpfile, redis

frameOk = True

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
    global frameOk
    while(frameOk):
        now = datetime.datetime.now()
        # Make sure thread doesn't hang in case of socket time out, etc.

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

def motionDetected(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout):
    """Actions to take after motion detected"""
    logger.info("Motion detected subprocess submit")
    return # remove to actually do something
    # SCP video file to central server
    thread = threading.Thread(target=scpfile.copyFile, args=(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout,))
    thread.start()
    logger.info("Motion detected subprocess submitted")

def pedestrianDetected(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout):
    """Actions to take after pedestrians detected"""
    logger.info("Pedestrian detected subprocess submit")
    return # remove to actually do something
    # SCP video file to central server
    thread = threading.Thread(target=scpfile.copyFile, args=(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout,))
    thread.start()
    logger.info("Pedestrian detected subprocess submitted")

def cascadeDetected(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout):
    """Actions to take after pedestrians detected"""
    logger.info("Cascade detected subprocess submit")
    return # remove to actually do something
    # SCP video file to central server
    thread = threading.Thread(target=scpfile.copyFile, args=(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout,))
    thread.start()
    logger.info("Cascade detected subprocess submitted")

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
    # Set SCP related attributes
    config.hostName = parser.get("scp", "hostName")
    config.userName = parser.get("scp", "userName")
    config.remoteDir = parser.get("scp", "remoteDir")
    config.timeout = parser.getint("scp", "timeout")
    config.deleteSource = parser.getboolean("scp", "deleteSource")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        configFileName = "motiondetect.ini"
    else:
        configFileName = sys.argv[1]
    parser = configparser.SafeConfigParser()
    # Read configuration file
    parser.read(configFileName)
    # Configure logger
    logger = logging.getLogger("motiondetect")
    logger.setLevel(parser.get("logging", "level"))
    formatter = logging.Formatter(parser.get("logging", "formatter"))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Load values from ini file
    config(parser)
    
    # See if we have MJPEG stream
    mjpeg = urllib.parse.urlparse(config.url).scheme == "http"
    # Initialize video
    if mjpeg:
        frameWidth, frameHeight = initMjpegVideo(config.url, config.socketTimeout)
        fps = config.fps
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
        # Init cascade classifier
        if config.detectType.lower() == "h":
            cascadedet.init(os.path.expanduser(config.cascadeFile))
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
        while(len(frameBuf) < fps):
            # 1/4 of FPS sleep
            print(len(frameBuf))
            time.sleep(3)              
        start = time.time()
        appstart = start
        print (appstart)
        
        #^ bullshit.
        r = redis.from_url(os.environ.get("REDIS_URL"))
        r.set('det_status', 'GO')
        # Loop as long as there are frames in the buffer
        while(r.get('det_status').decode('utf-8') == 'GO'):
            # Wait until frame buffer is full
            while(len(frameBuf) < fps):
                # 1/4 of FPS sleep
                time.sleep(3)
                
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

            
            
            
            if config.historyImage:
                # Update history image
                historyImg = numpy.bitwise_or(bwImg, historyImg)                    
            # Threshold to trigger motion
            if motionPercent > config.startThreshold or (recording and motionPercent >= config.stopThreshold):
                #if motionPercent >= config.maxChange:
                #    skipCount = config.skipFrames
                #    logger.debug("Maximum motion change: %4.2f" % motionPercent)
                if not recording:
                    #ONLYOCCURSONCE
                    # Construct directory name from camera name, recordDir and date
                    dateStr = now.strftime("%Y-%m-%d")
                    fileDir = "%s/%s/%s" % (os.path.expanduser(config.recordDir), config.cameraName, dateStr)
                    # Create dir if it doesn"t exist
                    if not os.path.exists(fileDir):
                        os.makedirs(fileDir)
                    fileName = "%s.%s" % (now.strftime("%H-%M-%S"), config.recordFileExt)
                    videoWriter = cv2.VideoWriter("%s/%s" % (fileDir, fileName), 
                                                  cv2.VideoWriter_fourcc(
                                                        config.fourcc[0], 
                                                        config.fourcc[1], 
                                                        config.fourcc[2], 
                                                        config.fourcc[3]), 
                                                  fps, 
                                                  (frameWidth, frameHeight), 
                                                  True
                                                  )
                    logger.info("Start recording (%4.2f) %s/%s @ %3.1f FPS" % (motionPercent, fileDir, fileName, fps))
                    recFrameNum = 1
                    peopleFound = False
                    cascadeFound = False
                    recording = True
                if config.mark:
                    # Draw rectangle around found objects
                    markRectSize(frame, movementLocations, widthMultiplier, heightMultiplier, (0, 255, 0), 2)
                # Detect pedestrians ?
                if config.detectType.lower() == "p":
                    locationsList, foundLocationsList, foundWeightsList = pedestriandet.detect(movementLocations, resizeImg, config.winStride, config.padding, config.scale0)
                    if len(foundLocationsList) > 0:
                        # Only filter if minWeight > 0.0
                        if config.minWeight > 0.0:
                            # Filter found location by weight
                            foundLocationsList, foundWeightsList = filterByWeight(foundLocationsList, foundWeightsList, config.minWeight)
                        # Any hits after possible filtering?
                        if len(foundLocationsList) > 0:
                            peopleFound = True
                            if config.mark:
                                # Draw rectangle around found objects
                                markRectWeight(frame, locationsList, foundLocationsList, foundWeightsList, widthMultiplier, heightMultiplier, (255, 0, 0), 2)
                            # Save off detected elapsedFrames
                            if config.saveFrames:
                                thread = threading.Thread(target=saveFrame, args=(frame, "%s/pedestrian-%s" % (fileDir, os.path.splitext(fileName)[0]), "%d.jpg" % recFrameNum,))
                                thread.start()
                            logger.debug("Pedestrian detected locations: %s" % foundLocationsList)
                # Haar Cascade detection?
                elif config.detectType.lower() == "h":
                    locationsList, foundLocationsList = cascadedet.detect(movementLocations, grayImg, config.scaleFactor, config.minNeighbors, config.minWidth, config.minHeight)
                    if len(foundLocationsList) > 0:
                        cascadeFound = True
                        if config.mark:
                            # Draw rectangle around found objects
                            markRoi(frame, locationsList, foundLocationsList, widthMultiplier, heightMultiplier, (255, 0, 0), 2)
                            # Save off detected elapsedFrames
                            if config.saveFrames:
                                thread = threading.Thread(target=saveFrame, args=(frame, "%s/cascade-%s" % (fileDir, os.path.splitext(fileName)[0]), "%d.jpg" % recFrameNum,))
                                thread.start()
                        logger.debug("Cascade detected locations: %s" % foundLocationsList)

            # If recording write frame and check motion percent
            if recording:
                if len(historyBuf) > 0:
                    # Write first image in history buffer (the oldest)
                    videoWriter.write(historyBuf[0][0])
                    recFrameNum += 1
                # Threshold to stop recording or empty frame buffer
                if motionPercent <= config.stopThreshold or len(frameBuf) == 0:
                    # Write off frame buffer skipping frame already written
                    logger.info("Writing %d frames of history buffer" % len(frameBuf))
                    for f in historyBuf[1:]:
                        videoWriter.write(f[0])
                    logger.info("Stop recording")
                    del videoWriter
                    # Rename video to show pedestrian found
                    if peopleFound:
                        os.rename("%s/%s" % (fileDir, fileName), "%s/pedestrian-%s" % (fileDir, fileName))
                        pedestrianDetected(logger, config.hostName, config.userName, "%s/pedestrian-%s" % (fileDir, fileName), "%s/%s" % (config.remoteDir, dateStr), config.deleteSource, config.timeout)
                    # Rename video to show cascade found
                    elif cascadeFound:
                        os.rename("%s/%s" % (fileDir, fileName), "%s/cascade-%s" % (fileDir, fileName))
                        cascadeDetected(logger, config.hostName, config.userName, "%s/cascade-%s" % (fileDir, fileName), "%s/%s" % (config.remoteDir, dateStr), config.deleteSource, config.timeout)
                    # Rename video to show motion found
                    else:
                        os.rename("%s/%s" % (fileDir, fileName), "%s/motion-%s" % (fileDir, fileName))
                        motionDetected(logger, config.hostName, config.userName, "%s/motion-%s" % (fileDir, fileName), "%s/%s" % (config.remoteDir, dateStr), config.deleteSource, config.timeout)
                    recording = False
        
        
        
        
        elapsed = time.time() - appstart
        logger.info("Calculated %4.1f FPS, elapsed time: %4.2f seconds, frame total: %s" % (frameTotal / elapsed, elapsed, frameTotal))
        # Exit video streaming thread
        frameOk = False
        # Clean up
        if mjpeg:
            socketFile.close()
            streamSock.close()
        else:
            del videoCapture
        if config.historyImage and fileDir is not None:
            # Save history image ready for ignore mask editing
            logger.info("%s/%s.png" % (fileDir, fileName))
            cv2.imwrite("%s/%s.png" % (fileDir, fileName), cv2.bitwise_not(historyImg))

if __name__ == '__main__':
    main()
