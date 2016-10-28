# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:02 2016

@author: Peter
"""

import cv2
import PyRealSense
from mayavi import mlab

#enums that need to go into lib
Depth = 0
IR = 1
Color = 2

comget = PyRealSense.getDepthLaserPower
comset = PyRealSense.setDepthLaserPower
comname = ' depth laser power'
cominc = 1

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
        try:
            frame = PyRealSense.getframe()
            depth = frame[0]
            depth = depth/depth.max()
           # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET) 
            
            cv2.namedWindow("Depth")
            cv2.imshow('Depth', depth)
            cv2.namedWindow("RGB")
            cv2.imshow('RGB', frame[2])
            cv2.namedWindow("IR")
            cv2.imshow('IR', frame[1])
    
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            if key & 0xFF == ord('w'):
                comget = PyRealSense.getDepthLaserPower
                comset = PyRealSense.setDepthLaserPower
                comname = "depth laser Power"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('e'):
                comget = PyRealSense.getDepthFilter
                comset = PyRealSense.setDepthFilter
                comname = "depth filter"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('r'):
                comget = PyRealSense.getDepthAccuracy
                comset = PyRealSense.setDepthAccuracy
                comname = "depth accuracy"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('t'):
                comget = PyRealSense.getDepthRange
                comset = PyRealSense.setDepthRange
                comname = "depth range"
                cominc = 5
                print([comname, comget()])
            if key & 0xFF == ord('p'):
                comget = PyRealSense.getColorBackLightCompensation
                comset = PyRealSense.setColorBackLightCompensation
                comname = "backlight comp"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('a'):
                comget = PyRealSense.getColorBrightness
                comset = PyRealSense.setColorBrightness
                comname = "brightness"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('s'):
                comget = PyRealSense.getColorContrast
                comset = PyRealSense.setColorContrast
                comname = "contrast"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('d'):
                comget = PyRealSense.getColorHue
                comset = PyRealSense.setColorHue
                comname = "hue"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('f'):
                comget = PyRealSense.getColorGamma
                comset = PyRealSense.setColorGamma
                comname = "gamma"
                cominc = 1
                print([comname, comget()])
            if key & 0xFF == ord('g'):
                comget = PyRealSense.getColorGain
                comset = PyRealSense.setColorGain
                comname = "gain"
                cominc = 1
                print([comname, comget()])


            


            if key & 0xFF == ord('y'):
                PyRealSense.setColorAutoExposure(not(PyRealSense.getColorAutoExposure()))
                print(["colour autoexposure", PyRealSense.getColorAutoExposure()])
            if key & 0xFF == ord('u'):
               PyRealSense.setColorAutoWhiteBalance(not(PyRealSense.getColorAutoWhiteBalance()))
               print(["colour auto whiteballance", PyRealSense.getColorAutoWhiteBalance()])
            if key & 0xFF == ord('i'):
               PyRealSense.setColorAutoPowerLineFrequency(not(PyRealSense.getColorAutoPowerLineFrequency()))
               print(["Auto power line frequency", PyRealSense.getColorAutoPowerLineFrequency()])     
            if key & 0xFF == ord('o'):
               PyRealSense.setMirrorMode(not(PyRealSense.getMirrorMode()))
               print(["Mirror mode", PyRealSense.getMirrorMode()])     
            

            if key & 0xFF == ord('z'):
                comset(comget() - cominc)
                print([comname, comget()])
            if key & 0xFF == ord('x'):
                comset(comget() + cominc)
                print([comname, comget()])

            if key & 0xFF == ord('/'):
                print(PyRealSense.getCal(0),)
                print(PyRealSense.getCal(1),"\n")
                print(PyRealSense.getCal(2),"\n")


        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)




    # When everything done, release the capture
    PyRealSense.reldev()
    cv2.destroyAllWindows()