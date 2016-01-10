# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:02 2016

@author: Peter
"""

import cv2
import spam
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

global  markerf

if __name__ == '__main__':
  

    spam.getdev()

        
    frame = spam.getframe()
        

    while True:
        
       # print frame[240,320,:].tolist()
       # cv2.line(frame,(240,318),(240,322),(0,0,255))
       # cv2.line(frame,(238,320),(242,320),(0,0,255))
        frame = frame/1000
        cv2.imshow('Test Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame = spam.getframe()

    # When everything done, release the capture
    spam.reldev()
    cv2.destroyAllWindows()
    