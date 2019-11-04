import numpy as np
import os.path as osp

def readGroundTruth(path):
    with open(path,'r') as f:
        lines = f.readlines()

    res = [float(x) for x in lines]
    return np.array(res)

def readInput(path):
    with open(path,'r') as f:
        lines = f.readlines()
    
    res = [x.split() for x in lines]
    res = [[float(x[0]),float(x[1])] for x in res]
    return np.array(res)

def get2ClassData():
    py = osp.join("data","exam_y.dat")
    px = osp.join("data","exam_x.dat")
    x = readInput(px)
    y = readGroundTruth(py)
    return x,y
def get3ClassData():
    py = osp.join("data","iris_y.dat")
    px = osp.join("data","iris_x.dat")
    x = readInput(px)
    y = readGroundTruth(py)
    return x,y

if __name__ == "__main__":
    x,y = get3ClassData()
    print(x.shape)
    print(y.shape)
