import os
import socket
import datetime

def write(msg):
    t = datetime.datetime.now().isoformat()
    host = socket.gethostname()
    pid = os.getpid()

    f = open("output/global.log", "a")
    f.write("{:s}, host: {:s}, pid: {:d}, {:s}\n".format(t, host, pid, msg))
    f.close()

