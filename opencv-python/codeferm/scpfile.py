"""
Copyright (c) Steven P. Goldsmith. All rights reserved.

Created by Steven P. Goldsmith on February 25, 2017
sgoldsmith@codeferm.com
"""

import os, subprocess

# We cache this, so remote dir is created only once
curRemoteDir = None

def copyFile(logger, hostName, userName, localFileName, remoteDir, deleteSource, timeout):
    """SCP file using command line."""
    global curRemoteDir
    # Create remote dir only once
    if curRemoteDir != remoteDir:
        curRemoteDir = remoteDir
        # Give mkdir time to work
        beforeScp = "sleep 2; "
        # mkdir on remote host
        command = "ssh %s@%s \"%s\"" % (userName, hostName, "mkdir -p %s" % remoteDir)
        logger.info(" Submitting %s" % command)
        proc = subprocess.Popen([command], shell=True)
        logger.info("Submitted process %d" % proc.pid)
    else:
        beforeScp = ""
    # scp file
    if deleteSource:
        deleteCommand = "; rm -f %s" % localFileName
    else:
        deleteCommand = ""
    command = "%sscp %s %s@%s:%s/%s%s" % (beforeScp, localFileName, userName, hostName, remoteDir, os.path.basename(localFileName), deleteCommand)
    logger.info(" Submitting %s" % command)
    proc = subprocess.Popen([command], shell=True)
    logger.info("Submitted process %d" % proc.pid)
