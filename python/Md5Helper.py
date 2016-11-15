from __future__ import print_function

import hashlib
import os
from binascii import crc32


def mkmd5frombinary(str):
    try:
        m = hashlib.new("md5")
        m.update(str)
    except:
        pass
    return m.hexdigest()


def mkmd5fromfile(filename, blockM=1):
    opened = False
    fobj = None
    if hasattr(filename, "read"):
        fobj = filename
    else:
        if os.path.exists(filename) and os.path.isfile(filename):
            fobj = open(filename, "rb")
            opened = True
    if fobj:
        blockB = blockM * 1024 * 1024
        try:
            m = hashlib.new("md5")
            while True:
                fb = fobj.read(blockB)
                if not fb:
                    break
                m.update(fb)
        finally:
            if opened:
                fobj.close()
        return m.hexdigest()
    else:
        return 0


def mkcrc32fromfile(filename, blockM=1):  # block with (M)
    if os.path.exists(filename) and os.path.isfile(filename):
        blockB = blockM * 1024 * 1024
        crc = 0
        f = open(filename, "rb")
        while True:
            fb = f.read(blockB)
            if not fb:
                break
            crc = crc32(fb, crc)
        f.close()
        res = ''
        for i in range(4):
            t = crc & 0xFF
            crc >>= 8
            res = '%02x%s' % (t, res)
        return "0x" + res
    else:
        return 0


def run_unittest():
    print(mkmd5frombinary("24d30c4e25a2dc6f4f7d8be89f002605020fd938"))

if __name__ == "__main__":
    run_unittest()
