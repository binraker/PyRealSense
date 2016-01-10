# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:30:14 2016

@author: Peter
"""
import numpy as np

def create_point_cloud(depth_image):
    shape = depth_image.shape;
    rows = shape[0];
    cols = shape[1];

    points = np.zeros((rows * cols, 3), np.float32);

    bytes_to_units = (1.0 );

    # Linear iterator for convenience
    i = 0
    # For each pixel in the image...
    for r in xrange(0, rows):
        for c in xrange(0, cols):
            # Get the depth in bytes
            depth = depth_image[r, c];

            # If the depth is 0x0 or 0xFF, its invalid.
            # By convention it should be replaced by a NaN depth.
            
            # The true depth of the pixel in units
            z = depth * bytes_to_units;

            # Get the x, y, z coordinates in units of the pixel
            points[i, 0] = c;
            points[i, 1] = r;
            points[i, 2] = z
            
            i = i + 1
    return points