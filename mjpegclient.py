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


"""def open(url, timeout):

    # Set socket timeout
    socket.setdefaulttimeout(timeout)
    # Parse URL
    parsed = urllib.parse.urlparse(url)
    port = parsed.port
    # Set port to default if not set
    if not port:
        port = 5000
    # See if query string present
    if not parsed.query:
        path = parsed.path
    else:
        path = "%s%s%s" % (parsed.path, "?", parsed.query)   
    print (path)
    # See if we need to do HTTP basic access authentication
    if parsed.username is None:
        # Build HTTP header
        lines = [
            "GET %s HTTP/1.1" % path,
            "Host: %s" % parsed.hostname,
        ]
    else:
        # Base64 encode username and password
        token = base64.encodestring("%s:%s" % (parsed.username, parsed.password)).strip()
        # Build HTTP header
        lines = [
            "GET %s HTTP/1.1" % path,
            "Host: %s" % parsed.hostname,
            "Authorization: Basic %s" % token,
        ]
    # AF_INET: IPv4 protocols (both TCP and UDP)
    # SOCK_STREAM: a connection-oriented, TCP byte stream
    streamSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    streamSock.connect((parsed.hostname, port))
    # Socket file in read, write, binary mode and no buffer
    socketFile = streamSock.makefile("rwb")
    # Send HTTP GET for MJPEG stream
    msg = "\r\n".join(lines) + "\r\n\r\n"
    socketFile.write(msg.encode())
    # Read in HTTP headers
    line = socketFile.readline()
    while len(line) > 0 and line.strip() != "":
        print("UIASDHIUSDHD")
        parts = line.split(":")
        if len(parts) > 1 and parts[0].lower() == "content-type":
            # Extract boundary string from content-type
            content_type = parts[1].strip()
            boundary = content_type.split(";")[1].split("=")[1]
        line = socketFile.readline()
    # See if we found "content-type"
    if not boundary:
        raise Exception("Cannot find content-type")
    return socketFile, streamSock, boundary

def getFrameLength(socketFile, boundary):

    line = socketFile.readline()
    # Find boundary
    while len(line) > 0 and line.count(boundary) == 0:
        print("UIAfdfddffdSDHD")
        line = socketFile.readline()
    # Read in chunk headers
    while len(line) > 0 and line.strip() != "":
        print("UasdadadsadaSDHIUSDHD")
        parts = line.split(":")
        if len(parts) > 1 and parts[0].lower().count("content-length") > 0:
            # Grab chunk length
            length = int(parts[1].strip())
        line = socketFile.readline()
    return length"""
    
def get_image(r):
    s3 = boto3.client('s3')
    s3.download_file(os.environ.get("S3_BUCKET"), '360_stream/stream.jpg', 'stream.jpg')
    
    r.set('det_ready', False)
    
    return cv2.imread('stream.jpg')    
    
    

def getFrame():
    """Get raw frame data from stream and decode"""
    r = redis.from_url(os.environ.get("REDIS_URL"))

    while r.get('det_ready').decode('utf-8') == "False":
        time.sleep(5)
    print("Getting Image")
    frame = get_image(r)
    
    #jpeg = socketFile.read(getFrameLength(socketFile, boundary))
    #return jpeg, cv2.imdecode(numpy.fromstring(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    
    
    return frame
    
