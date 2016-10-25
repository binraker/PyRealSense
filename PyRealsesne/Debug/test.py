# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:02 2016

@author: Peter
"""

import cv2
import PyRealSense

global  markerf
Lp=16
Filter=5
Acc = 0
range = 50
if __name__ == '__main__':
  
    PyRealSense.getdev()
    
    print(PyRealSense.getDepthLaserPower())
    print(PyRealSense.getDepthFilter())
    print(PyRealSense.getDepthAccuracy())
    print(PyRealSense.getDepthRange())
    print(PyRealSense.getColorAutoExposure())
    print(PyRealSense.getColorAutoPowerLineFrequency())
    print(PyRealSense.getColorAutoWhiteBalance())
    print(PyRealSense.getColorBackLightCompensation())
    print(PyRealSense.getColorBrightness())
    print(PyRealSense.getColorContrast())
    print(PyRealSense.getColorExposure())
    print(PyRealSense.getColorHue())
    print(PyRealSense.getColorFieldOfView())
    print(PyRealSense.getColorFocalLength())
    print(PyRealSense.getColorFocalLengthMM())
    print(PyRealSense.getColorGain())
    print(PyRealSense.getColorGamma())
    print(PyRealSense.getColorPowerLineFrequency())
    print(PyRealSense.getColorPrincipalPoint())
    print(PyRealSense.getColorSaturation())
    print(PyRealSense.getColorSharpness())
    print(PyRealSense.getColorWhiteBalance())
    print(PyRealSense.getDepthConfidenceThreshold())
    print(PyRealSense.getDepthFieldOfView())
    print(PyRealSense.getDepthFocalLength())
    print(PyRealSense.getDepthFocalLengthMM())
    print(PyRealSense.getDepthLowConfidenceValue())
    print(PyRealSense.getDepthPrincipalPoint())
    print(PyRealSense.getDepthSensorRange())
    print(PyRealSense.getDeviceAllowProfileChange())
    print(PyRealSense.getDeviceAllowProfileChange())
    print(PyRealSense.getDepthUnit())
    print(PyRealSense.getMirrorMode())
    while True:         
        
        frame = PyRealSense.getframe()
        
        cv2.namedWindow("Depth")
        cv2.imshow('Depth', frame[0]/1000)
        cv2.namedWindow("RGB")
        cv2.imshow('RGB', frame[2])
        cv2.namedWindow("IR")
        cv2.imshow('IR', frame[1])
    
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('s'):
            PyRealSense.setDepthLaserPower(PyRealSense.getDepthLaserPower() - 1)
        if key & 0xFF == ord('w'):
            PyRealSense.setDepthLaserPower(PyRealSense.getDepthLaserPower() + 1)

        if key & 0xFF == ord('d'):
            PyRealSense.setDepthFilter(PyRealSense.getDepthFilter() - 1)
        if key & 0xFF == ord('e'):
            PyRealSense.setDepthFilter(PyRealSense.getDepthFilter() + 1)

        if key & 0xFF == ord('f'):
            PyRealSense.setDepthAccuracy(PyRealSense.getDepthAccuracy() - 1)
        if key & 0xFF == ord('r'):
            PyRealSense.setDepthAccuracy(PyRealSense.getDepthAccuracy() + 1)

        if key & 0xFF == ord('g'):
            PyRealSense.setDepthRange(PyRealSense.getDepthRange() - 1)
        if key & 0xFF == ord('t'):
            PyRealSense.setDepthRange(PyRealSense.getDepthRange() + 1)

        if key & 0xFF == ord('p'):
            print ([Lp,Filter, Acc])


    # When everything done, release the capture
    PyRealSense.reldev()
    cv2.destroyAllWindows()