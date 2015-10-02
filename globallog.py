import os
import socket
import datetime

def write(msg):
    t = datetime.datetime.now().isoformat()
    host = socket.gethostname()
    pid = os.getpid()

    f = open("output/global.log", "a")
    f.write("{0:s}, host: {1:s}, pid: {2:d}, {3:s}\n".format(t, host, pid, msg))
    f.close()

