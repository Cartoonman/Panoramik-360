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
import pedestriandet
import pickle


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



def restoreScale(loclist, widthMultiplier, heightMultiplier):
    x = list(map(lambda x: (x[0] * widthMultiplier, x[1] * heightMultiplier, x[2] * widthMultiplier, x[3] * heightMultiplier), loclist))
    return x


def markRectWeight(target, locList, foundLocsList, foundWghtsList, boxColor, boxThickness):
    """Mark ROI rectangles with weight in image"""
    for location, foundLocations, foundWeights in zip(locList, foundLocsList, foundWghtsList):
        i = 0
        # Mark target
        for x, y, w, h in foundLocations:
            # Calculate full size
            x2 = x 
            y2 = y 
            w2 = w 
            h2 = h             
            x3, y3, w3, h3 = location
            # Calculate full size
            x4 = x3 
            y4 = y3 
            # Mark target
            cv2.rectangle(target, (x2 + x4, y2 + y4), (x2 + x4 + w2, y2 + y4 + h2), boxColor, boxThickness)
            label = "%1.2f" % foundWeights[i]
            # Figure out text size
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]            
            # Print weight
            cv2.putText(target, label, (x2 + x4, y2 + y4 + h2 - size[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            i += 1





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
    cv2.imwrite('/tmp/frames/imgDILATE'+str(indx) + '.jpg', image)
    # The bright areas of the image (the background, apparently), get thinner, whereas the dark zones bigger
    image = cv2.erode(image, None, iterations=erodeAmount);
    cv2.imwrite('/tmp/frames/imgERODE'+str(indx) + '.jpg', image)
    # Find contours
    image, contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Add objects with motion
    movementLocations = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        movementLocations.append(rect)
    return movementLocations
    
indx = 0
def detect(movingAvgImg, maskImg, image, kSize, alpha, blackThreshold, maxChange, dilateAmount, erodeAmount, detectType, hitThreshold, winStride, padding, scale0, minWeight, widthMultiplier, heightMultiplier, redis):
    global indx
    """Detect motion""" 
    movementLocationsFiltered = []
    # Generate work image by blurring
    workImg = cv2.blur(image, kSize)
    # Generate moving average image if needed
    if movingAvgImg is None:
        movingAvgImg = numpy.float32(workImg)
    # Generate moving average image
    cv2.accumulateWeighted(workImg, movingAvgImg, alpha)
    diffImg = cv2.absdiff(workImg, cv2.convertScaleAbs(movingAvgImg))
    # Convert to grayscale
    grayImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
    # Convert to BW
    ret, bwImg = cv2.threshold(grayImg, blackThreshold, 255, cv2.THRESH_BINARY)
    # Apply ignore mask
    if maskImg is not None:
        bwImg = numpy.bitwise_and(bwImg, maskImg)     
    # Total number of changed motion pixels
    height, width, unknown = image.shape
    motionPercent = 100.0 * cv2.countNonZero(bwImg) / (width * height)
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
    # Pedestrian detection disabled for now   
    """if detectType.lower() == "p":
        locationsList, foundLocationsList, foundWeightsList = pedestriandet.detect(movementLocations, 
                                                                                    image, 
                                                                                    winStride, 
                                                                                    padding, 
                                                                                    scale0)
        if len(foundLocationsList) > 0:
            # Only filter if minWeight > 0.0
            if minWeight > 0.0:
                # Filter found location by weight
                foundLocationsList, foundWeightsList = filterByWeight(foundLocationsList, 
                                                                      foundWeightsList, 
                                                                      minWeight)
            # Any hits after possible filtering?
            if len(foundLocationsList) > 0:
                peopleFound = True
                # Draw rectangle around found objects
                markRectWeight(video_image, locationsList, foundLocationsList, foundWeightsList, (255, 0, 0), 2)
                # Save off detected elapsedFrames
                print("Pedestrian detected locations: %s" % foundLocationsList)"""                
                
       
    markRectSize(video_image, movementLocationsFiltered, (0, 255, 0), 2)
    markRectSize(workImg, movementLocationsFiltered, (0, 255, 0), 2)
    video_movingAvgImg = movingAvgImg.copy()
    markRectSize(video_movingAvgImg, movementLocationsFiltered, (0, 255, 0), 2)
    markRectSize(diffImg, movementLocationsFiltered, (0, 255, 0), 2)
    video_grayImg = cv2.cvtColor(grayImg.copy(),cv2.COLOR_GRAY2RGB)
    markRectSize(video_grayImg, movementLocationsFiltered, (0, 255, 0), 2)
    video_bwImg = cv2.cvtColor(bwImg.copy(),cv2.COLOR_GRAY2RGB)
    markRectSize(video_bwImg, movementLocationsFiltered, (0, 255, 0), 2)
    cv2.imwrite('/tmp/frames/imgw'+str(indx) + '.jpg', workImg)
    cv2.imwrite('/tmp/frames/img'+ str(indx) + '.jpg', video_image)
    cv2.imwrite('/tmp/frames/imgma'+str(indx) + '.jpg', video_movingAvgImg)
    cv2.imwrite('/tmp/frames/imgd'+str(indx) + '.jpg', diffImg)
    cv2.imwrite('/tmp/frames/imgmaG'+str(indx) + '.jpg', video_grayImg)
    cv2.imwrite('/tmp/frames/imgmaBW'+str(indx) + '.jpg', video_bwImg)
    # Detect if camera is adjusting and reset reference if more than threshold
    if motionPercent > maxChange:
        movingAvgImg = numpy.float32(workImg)
    
    indx = indx + 1
    
    p_mydict = pickle.dumps(restoreScale(movementLocationsFiltered, widthMultiplier, heightMultiplier))
    redis.set('DATA', p_mydict)
    
    return movingAvgImg, grayImg, bwImg, motionPercent, movementLocationsFiltered
