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
        lines.append(points)
        azimuth += azimuth_precision

    return lines
        
def height_terrarium(map_arr, x, y):
    # meters = (red * 256 + green + blue / 256) - 32768
    r, g, b = map_arr[y, x]
    
    # r, g, b = r*256, g*256, b # map from (0-1) to (0-256) # 2.23s
    # meters = r*256 + g + b - 32768
    
    meters = 256*(256*r + g - 128) + b # 1.58s
    # meters = 65536*r + 256*b + b - 32768 # 2.08 sec
    

    return meters
    pass

def transform_map(map_arr):
    #f = lambda x: x[0]*256 + x[1] + x[2]/256
    
    r = map_arr[:,:, 0] * 65536
    g = map_arr[:,:, 1] * 256
    b = map_arr[:,:, 2]
    t = r + b + g - 32768
    return t

def ridge2(map_arr, viewpoint):
    
    start = time.time()
    hmap = transform_map(map_arr)
    print(f'create hmap time: {time.time() - start}')
    (height, width, _ ) = map_arr.shape
    # strategy 1 choose all edge pixels of map as target
    blines = lines(viewpoint[0], viewpoint[1], width, height, 4, 2*math.pi/360, -math.pi, math.pi)
    
    ridge_points = []
    bresenttime = 0
    hlinetime = 0
    ridgeptime = 0
    for bline in blines:
        # start = time.time()
        # bline = bham.bresenham(viewpoint, target)
        # bresenttime += time.time()-start

        start = time.time()
        #height_line = [height_terrarium(map_arr, x, y) for (x, y) in bline]
        height_line = [hmap[y, x] for (x,y) in bline]

        hlinetime += time.time()-start

        start = time.time()
        ridge_point_index, _ = ridgepoint(2, height_line)
        ridgeptime += time.time()-start

        
        ridge_points.append(bline[ridge_point_index] + (height_line[ridge_point_index],))

    print(f'bline: {bresenttime}\nhline: {hlinetime}\nridgep: {ridgeptime}')
    
    return ridge_points

def ridge(map_arr, viewpoint):
    
    start = time.time()
    hmap = transform_map(map_arr)
    print(f'create hmap time: {time.time() - start}')
    (height, width, _ ) = map_arr.shape
    # strategy 1 choose all edge pixels of map as target
    targets = edge_points(width, height)
    
    ridge_points = []
    bresenttime = 0
    hlinetime = 0
    ridgeptime = 0
    for target in targets[::1]:
        start = time.time()
        bline = bham.bresenham(viewpoint, target)
        bresenttime += time.time()-start

        start = time.time()
        #height_line = [height_terrarium(map_arr, x, y) for (x, y) in bline]
        height_line = [hmap[y, x] for (x,y) in bline]

        hlinetime += time.time()-start

        start = time.time()
        ridge_point_index, _ = ridgepoint(2, height_line)
        ridgeptime += time.time()-start

        
        ridge_points.append(bline[ridge_point_index] + (height_line[ridge_point_index],))

    print(f'bline: {bresenttime}\nhline: {hlinetime}\nridgep: {ridgeptime}')
    
    return ridge_points

def cartesian_to_cylindrical(center, point):
    (x0, y0, z0) = center
    x, y, z = point
    azimuth = math.atan2(y0-y, x-x0)
    radius = math.sqrt((y - y0)**2 + (x - x0)**2 + (z - z0)**2)
    angle = math.asin((z - z0) / radius)
    return azimuth, angle
    

    
def pxpoint_to_meters(mpp, p):
    x, y, z = p
    return x*mpp, y*mpp, z

def main():
    lat, lon = 35.35813931744461, 138.63260800348849
    lat, lon = 45.877630, 10.857161
    zoom = 10
    radius = 0.2
    start = time.time()
    map = mp.create_map(lat, lon, zoom, radius)
    map_arr = map.get('map')
    print(f'create map time: {time.time() - start }')

    start = time.time()
    (height, width, _ ) = map_arr.shape
    viewpoint = (width // 2, height // 2)
    viewpoint_height = height_terrarium(map_arr, viewpoint[0], viewpoint[1])
    rpoints = ridge2(map_arr, viewpoint)
    x, y, height = zip(*rpoints)
    print(viewpoint, viewpoint_height)
    mpp = map.get('mpp')
    x_azi = []
    y_ang = []
    for p in rpoints:
        azi, angle = cartesian_to_cylindrical(
            pxpoint_to_meters(mpp, viewpoint + (viewpoint_height,)), 
            pxpoint_to_meters(mpp, p))
        x_azi.append(azi)
        y_ang.append(angle)
    
    cyl_points = zip(x_azi, y_ang)
    cyl_points = sorted(cyl_points, key=lambda p : p[0])
    x_azi_s, y_ang_s = zip(*cyl_points)
    
    print(f'create ridge time: {time.time() - start }')
    
    fig, (ax1, ax2) = plt.subplots(2)
    
    plt.imshow(map_arr)
    ax2.plot(x, y, linewidth=2, color='magenta')
    ax1.plot(x_azi_s, y_ang_s)
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