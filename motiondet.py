"""
Copyright (c) Steven P. Goldsmith. All rights reserved.

Created by Steven P. Goldsmith on February 8, 2017
sgoldsmith@codeferm.com
"""

"""Motion detector using moving average.

@author: sgoldsmith

"""

import numpy, cv2
import os




def markRectSize(target, rects, boxColor, boxThickness):
    """Mark rectangles in image"""
    for x, y, w, h in rects:
        # Calculate full size
        x2 = x 
        y2 = y 
        w2 = w 
        h2 = h 
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
        
        

def inside(r, q):
    """See if one rectangle inside another"""
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def contours(image, dilateAmount, erodeAmount):
    """Return contours"""
    # The background (bright) dilates around the black regions of frame
    image = cv2.dilate(image, None, iterations=dilateAmount);
    # The bright areas of the image (the background, apparently), get thinner, whereas the dark zones bigger
    image = cv2.erode(image, None, iterations=erodeAmount);
    # Find contours
    image, contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Add objects with motion
    movementLocations = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        movementLocations.append(rect)
    return movementLocations
indx = 0
def detect(movingAvgImg, maskImg, image, kSize, alpha, blackThreshold, maxChange, dilateAmount, erodeAmount):
    global indx
    """Detect motion"""
    
    
    
    movementLocationsFiltered = []
    # Generate work image by blurring
    workImg = cv2.blur(image, kSize)
    cv2.imwrite('/tmp/frames/imgw'+str(indx) + '.jpg', workImg)
    # Generate moving average image if needed
    if movingAvgImg is None:
        movingAvgImg = numpy.float32(workImg)
    # Generate moving average image
    cv2.accumulateWeighted(workImg, movingAvgImg, alpha)
    cv2.imwrite('/tmp/frames/imgma'+str(indx) + '.jpg', movingAvgImg)
    diffImg = cv2.absdiff(workImg, cv2.convertScaleAbs(movingAvgImg))
    cv2.imwrite('/tmp/frames/imgd'+str(indx) + '.jpg', diffImg)
    # Convert to grayscale
    grayImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/tmp/frames/imgmaG'+str(indx) + '.jpg', grayImg)
    # Convert to BW
    ret, bwImg = cv2.threshold(grayImg, blackThreshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite('/tmp/frames/imgmaBW'+str(indx) + '.jpg', bwImg)
    # Apply ignore mask
    if maskImg is not None:
        bwImg = numpy.bitwise_and(bwImg, maskImg)     
    # Total number of changed motion pixels
    height, width, unknown = image.shape
    motionPercent = 100.0 * cv2.countNonZero(bwImg) / (width * height)
    # Detect if camera is adjusting and reset reference if more than threshold
    if motionPercent > maxChange:
        movingAvgImg = numpy.float32(workImg)
    movementLocations = contours(bwImg, dilateAmount, erodeAmount)
    
    # Filter out inside rectangles
    for ri, r in enumerate(movementLocations):
        for qi, q in enumerate(movementLocations):
            if ri != qi and inside(r, q):
                break
        else:
            rx, ry, rw, rh = r
            regPercent = ((rw * rh) / (width * height)) * 100.0
            # Toss rectangles >= maxChange percent of total frame
            if regPercent < maxChange :
                movementLocationsFiltered.append(r)
    video_image = image.copy()
    markRectSize(video_image, movementLocationsFiltered, (0, 255, 0), 2)
    cv2.imwrite('/tmp/frames/img'+ str(indx) + '.jpg', video_image)
    indx = indx + 1
    return movingAvgImg, grayImg, bwImg, motionPercent, movementLocationsFiltered
