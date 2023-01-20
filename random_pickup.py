import os
import shutil
import random
import argparse

def imagesList(oriDir):
    files = []
    for r, d ,f in os.walk(oriDir):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def randCopy(oriDir, dstDir, num):
    fileList = imagesList(oriDir)
    random.shuffle(fileList)
    for idx, f in enumerate(fileList):
        if(len(os.listdir(dstDir)) == num): break
        shutil.copy(f, dstDir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input Dir", required=True, type=str)
    parser.add_argument("--output", help="Output Dir", required=True, type=str)
    parser.add_argument("amount", help="Amount", metavar='N', type=int)
    arg = parser.parse_args()
    randCopy(arg.input, arg.output, arg.amount) 
