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
        
    spam.getframe()
    frame = spam.getdepth() 
    framec = frame.copy()

    while True:

        framec = frame/1000  
        cv2.imshow('Test Frame', framec)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        spam.relframe()    
        del frame
        del framec
        spam.getframe()

        frame = spam.getdepth()
        framec = frame.copy()


    # When everything done, release the capture
    spam.relframe()
    spam.reldev()
    cv2.destroyAllWindows()