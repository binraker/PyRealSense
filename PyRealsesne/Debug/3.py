# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:02 2016

@author: Peter
"""

import cv2
import spam
import numpy as np
global  markerf

if __name__ == '__main__':
  
    spam.getdev()
    depth = np.empty([480,640], dtype = np.float)
    ir = np.empty([480,640], dtype = np.uint8)
    rgb = np.empty([480,640,3], dtype = np.uint8)

    while True:         
        
        spam.getframe2(depth,ir,rgb)
        
        cv2.namedWindow("Depth")
        cv2.imshow('Depth', depth)
        cv2.namedWindow("RGB")
        cv2.imshow('RGB', ir)
        cv2.namedWindow("IR")
        cv2.imshow('IR', rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       # spam.relframe()
    # When everything done, release the capture
    spam.reldev()
    cv2.destroyAllWindows()