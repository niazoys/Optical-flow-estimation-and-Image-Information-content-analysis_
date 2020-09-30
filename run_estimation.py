#https://docs.opencv.org/4.4.0/d6/d00/tutorial_py_root.html
#https://docs.opencv.org/4.4.0/dd/d43/tutorial_py_video_display.html "Playing Video from file"

from __future__ import print_function

import argparse
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os


def computeMSE(prev, curr):
    mse = 0
    N=prev.shape[0]
    M=prev.shape[1] 
    mse=np.sum((prev-curr)**2)/(M*N)
   
    return mse
    
def computePSNR(mse):
    psnr = 0
    psnr= 10 * math.log10( (255**2)/ mse )
    return psnr

def computeEntropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])/(img.shape[0]*img.shape[1]) + 1e-4
    ent=-1*np.sum(hist *np.log2(hist))
    return ent
    
def computeErrorImage(im1, im2):
    res=cv2.min(255,cv2.max(0,im1-(im2+128)))
    return res


def computeOpticalFlow1(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res

    
def computeGME(flow):
    src = np.zeros_like(flow)
    
    h, w = flow.shape[:2]
    
    c = np.array([w/2, h/2])
    
    src[:,:,0] += np.arange(w)

    src[:,:,1] += np.arange(h)[:,np.newaxis]
    
    src -= c;

    srcPts = src.reshape((h*w, 2)) 
    
    dst=src+flow
    
    dstPts = dst.reshape((h*w, 2)) 
    
    homography,mask=cv2.findHomography(srcPts,dstPts,cv2.RANSAC)
    
    result=cv2.perspectiveTransform(src,homography)

    #TODO
    # You should use
    # - cv2.findHomography 
    #   see https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    # - cv2.perspectiveTransform
    #   see https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
    gme = result-src  #change this...

    return gme

def computeGMEError(flow, gme):
    #TODO
    err = (flow-gme) #change this
    return err[:,:,0]

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Read video file')
    parser.add_argument('video', help='input video filename')
    parser.add_argument('deltaT', help='input deltaT between frames', type=int)

    args = parser.parse_args()
    
    path=args.video
    directory=path.split(".")[0]
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("ERROR: unable to open video: "+args.video)
        quit()

    deltaT=args.deltaT
    previousFrames=[]
    frameNumbers = []
    mses = []
    psnrs = []
    mse0s = []
    psnr0s = []
    ents = []
    entEs = []

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if (ret==False):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if (len(previousFrames) >= deltaT):
            prev = previousFrames.pop(0)

            flow = computeOpticalFlow1(prev, gray)
        
        
            compensatedFrame = computeCompensatedFrame(prev, flow)

            cv2.imshow('compensated', compensatedFrame)
        
            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)
            
            cv2.imshow('imErr0', imErr0)
            cv2.imshow('imErr', imErr)

            
            mse0 = computeMSE(prev, gray)
            psnr0 = computePSNR(mse0)
            mse = computeMSE(compensatedFrame, gray)
            psnr = computePSNR(mse)
            ent = computeEntropy(gray)
            entE = computeEntropy(imErr)
        
            frameNumbers.append(i)
            mses.append(mse)
            psnrs.append(psnr)
            mse0s.append(mse0)
            psnr0s.append(psnr0)
            ents.append(ent)
            entEs.append(entE)
        
            
            gme = computeGME(flow)
            
            gmeError = computeGMEError(flow, gme)
            
            cv2.imshow('flow', draw_flow(gray, flow))
            cv2.imshow('gme', draw_flow(gray, gme))
            cv2.imshow('gmeError', gmeError)
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(os.path.join(directory , 'frame_flow'+str(i)+'.jpg'), draw_flow(gray, flow))
            cv2.imwrite(os.path.join(directory , 'frame_flow_imerr'+str(i)+'.jpg'), imErr)
            cv2.imwrite(os.path.join(directory , 'frame_flow_imerr0'+str(i)+'.jpg'), imErr0)

        
        previousFrames.append(gray.copy())
        i+=1
        
        cv2.imshow('frame', gray)
        
        cv2.waitKey(1)

    
    plt.plot(frameNumbers, mse0s, label='MSE0')
    plt.plot(frameNumbers, mses, label='MSE')
    plt.xlabel('frames')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE0 vs MSE')
    plt.savefig(str(directory)+"mse.png")
    plt.show()

    plt.plot(frameNumbers, ents, label='Entropy)')
    plt.plot(frameNumbers, entEs, label='EntropyE')
    plt.xlabel('frames')
    plt.ylabel('Entropy')
    plt.legend()
    plt.title('Entropy vs EntropyE')
    plt.savefig(str(directory)+"entropy.png")
    plt.show()    
    
    plt.plot(frameNumbers, psnr0s, label='PSNR0')
    plt.plot(frameNumbers, psnrs, label='PSNR')
    plt.xlabel('frames')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR0 vs PSNR')
    plt.savefig(str(directory)+"psnr.png")
    plt.show()
    
    
    cap.release()
    cv2.destroyAllWindows()