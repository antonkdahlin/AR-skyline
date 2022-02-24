from cmath import asin
from main import bresenham
import map as mp
import numpy as np
from matplotlib import image
import bresenham as bham
import math
import time

from matplotlib import pyplot as plt

def ridgepoint(height, line, printout = False):
    y0 = line[0] + height
    cum = -math.inf
    angle = None

    maxi = None
    if printout: print('dx\tdy\tangle\tcum')
    for dx, y in enumerate(line[1:], 1):
        dy = y - y0
        if cum < dy:
            angle = dy/dx
            cum = dy
            maxi = dx
        cum += angle


        if printout: print(f'{dx}\t{dy}\t{angle}\t{cum}')
    
    return (maxi, angle)


# points along rectangle defined by arguments. 
# points ordered clockwise starting topleft
def edge_points(width, height):
    top = [(x, 0) for x in range(width - 1)]
    right = [(width - 1, y) for y in range(0, height - 1)]
    bottom = [(x, height - 1) for x in range(width - 1, 0, -1)]
    left = [(0, y) for y in range(height - 1, 0, -1)]

    return top + right + bottom + left

def lines(x0, y0, width, height, line_precision, azimuth_precision, azimuth_start, azimuth_end):
    '''low line_precision make line more precise ex value of 1 move pointer one pixel at a time
    low azimuth_precision = more lines. ex value of 2pi/360 generates 360 lines'''
    azimuth = azimuth_start
    lines = []
    while azimuth < azimuth_end:
        
        dx = math.cos(azimuth) * line_precision
        dy = math.sin(azimuth) * line_precision
        
        x, y = x0, y0
        points = []
        while 0 <= x < width and 0 <= y < height:
            points.append((int(x), int(y)))
            x += dx
            y += dy
        lines.append((points, azimuth))
        azimuth += azimuth_precision

    return lines
        
def height_terrarium(map_arr, x, y):
    # meters = (red * 256 + green + blue / 256) - 32768
    r, g, b = map_arr[y, x]
    r, g, b = 255*r, 255*g, 255*b
    # r, g, b = r*256, g*256, b # map from (0-1) to (0-256) # 2.23s
    # meters = r*256 + g + b - 32768
    
    meters = 256*r + g + b/256 - 32768 # 1.58s
    # meters = 65536*r + 256*b + b - 32768 # 2.08 sec
    

    return meters
    pass

def transform_map(map_arr):
    #f = lambda x: x[0]*256 + x[1] + x[2]/256
    
    r = map_arr[:,:, 0] * 65280
    g = map_arr[:,:, 1] * 255
    b = map_arr[:,:, 2] * 0.99609375
    t = r + b + g - 32768
    return t

def ridge(hmap, viewpoint, height_offset, line_precision, azimuth_precision, azimuth_start, azimuth_end):
    (height, width) = hmap.shape
    x0, y0 = viewpoint
    
    rays = lines(x0, y0, width, height, line_precision, azimuth_precision, azimuth_start, azimuth_end)
    
    ridge_points = []
    for ray, azimuth in rays:
        height_line = [hmap[y, x] for (x,y) in ray]
        ridge_point_index, _ = ridgepoint(height_offset, height_line) 
        
        point = ray[ridge_point_index] + (height_line[ridge_point_index],)
        ridge_points.append(point)

    return ridge_points

def ridge2(hmap, viewpoint, height_offset, line_precision, azimuth_precision, azimuth_start, azimuth_end):
    (height, width) = hmap.shape
    x0, y0 = viewpoint
    
    rays = lines(x0, y0, width, height, line_precision, azimuth_precision, azimuth_start, azimuth_end)
    
    ridge_points = []
    for ray, azimuth in rays:
        height_line = [hmap[y, x] for (x,y) in ray]
        ridge_point_index, _ = ridgepoint(height_offset, height_line) 
        point = ray[ridge_point_index] + (height_line[ridge_point_index],)
        ridge_points.append((point, azimuth))

    return ridge_points


def cartesian_to_cylindrical(center, point):
    (x0, y0, z0) = center
    x, y, z = point
    azimuth = math.atan2(y-y0, x-x0)
    radius = math.sqrt((y - y0)**2 + (x - x0)**2 + (z - z0)**2)
    angle = math.asin((z - z0) / radius)
    return azimuth, angle
    

    
def pxpoint_to_meters(mpp, p):
    x, y, z = p
    return x*mpp, y*mpp, z





def main():
    # setup variables
    lat, lon = 35.35813931744461, 138.63260800348849 # mt fuji
    #lat, lon = 45.877630, 10.857161 # italy
    lat, lon = 45.8784571, 10.8567149
    lat, lon = 68.349389, 18.821194
    lat, lon = 68.271139, 18.974528
    # lat, lon = 65.5869998, 22.1494364 # luleå
    zoom = 10
    radius = 0.04
    height_offset = 2
    line_precision = 1
    azimuth_precision = 2*math.pi/720
    azimuth_start = -math.pi
    azimuth_end = math.pi 
    
    # get map for that area
    map = mp.create_map(lat, lon, zoom, radius)
    
    map_arr = map.get('map')
    mp.save_map_as_greyscale(map_arr, 'luleå.png')
    (height, width, _ ) = map_arr.shape

    viewpoint = (width // 2, height // 2)
    print(viewpoint)
    viewpoint_height = height_terrarium(map_arr, viewpoint[0], viewpoint[1])


    hmap = transform_map(map_arr) # transform to map with decoded height
    # rpoints = ridge(
    #     hmap, viewpoint, 
    #     height_offset, line_precision,
    #     azimuth_precision, azimuth_start, 
    #     azimuth_end)

    rpoints = ridge2(
        hmap, viewpoint, 
        height_offset, line_precision,
        azimuth_precision, azimuth_start, 
        azimuth_end)

    # for p in rpoints:
    #     print(p)
    rpoints, azimuth = zip(*rpoints)
    (x, y, height) = zip(*rpoints)


    mpp = map.get('mpp')
    rpoints_cyl = []
    for p in rpoints:
        azi, angle = cartesian_to_cylindrical(
            pxpoint_to_meters(mpp, viewpoint + (viewpoint_height,)), 
            pxpoint_to_meters(mpp, p))
        
        rpoints_cyl.append((azi, angle))

    # for p in rpoints_cyl:
    #     print(p)

    azi, ang = zip(*rpoints_cyl)
    
    fig, (ax1, ax2) = plt.subplots(2)
    
    plt.imshow(map_arr)
    ax2.scatter(x, y, s=10, color='r')
    ax1.plot(azimuth, ang)
    plt.show()

def test_lines():
    lines_ = lines(32,32,64,64,5,math.pi*2/100)

    fig, ax = plt.subplots()
    
    
    

    for line in lines_:
        x, y = zip(*line)
        ax.scatter(x, y)

    plt.show()
    

def test_edge_points():
    assert edge_points(2, 2) == [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert edge_points(3, 2) == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
    

if __name__ == "__main__":
    #test_lines()
    test_edge_points()
    main()