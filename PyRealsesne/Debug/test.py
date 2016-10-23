# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:02 2016

@author: Peter
"""

import cv2
import spam

global  markerf

if __name__ == '__main__':
  
    spam.getdev()

    while True:         
        
        frame = spam.getframe()
        
        cv2.namedWindow("Depth")
        cv2.imshow('Depth', frame[0]/1000)
        cv2.namedWindow("RGB")
        cv2.imshow('RGB', frame[2])
        cv2.namedWindow("IR")
        cv2.imshow('IR', frame[1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       # spam.relframe()
    # When everything done, release the capture
    spam.reldev()
    cv2.destroyAllWindows()