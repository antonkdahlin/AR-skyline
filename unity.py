import json
from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from ridge import ridge2, cartesian_to_cylindrical
from edgeextract import extract_skyline_with_preprocessing
import math


def main():
    low_vfov = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637872826261823552.json'
    vfov90 = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637872820052461187.json'
    res = importJson(low_vfov)

    # get skyline from image analysis
    imgpath = res.get('absoluteImgPath')
    img, short_line = extract_skyline_with_preprocessing(imgpath)

    # generate skyline from raster
    raster = raster_map_from_flat(res.get('raster_map'), res.get('grid_size_x'), res.get('grid_size_z'))
    #drawraster(raster)
    
    print(res.get("camera_worldPosition").get('x'))
    cam_pos = res.get("camera_worldPosition")
    ground_extents = res.get("ground_extents")
    ground_center = res.get("ground_center")
    cell_size = res.get("raster_cellsize")

    cam_xz_t = transform_coord(cam_pos.get('x'), cam_pos.get('z'),
        ground_center.get('x'), ground_center.get('z'),
        ground_extents.get('x'), ground_extents.get('z'),
        cell_size)
    
    rpoints = ridge2(raster, cam_xz_t, cam_pos.get('y'), 1, 2*math.pi/360, -math.pi, math.pi)
    print(f'cam_pos: {cam_xz_t + (cam_pos.get("y"),)}')
    # for p in rpoints:
    #     p, dir = p
    #     if -2.1 < dir  < -1.9:
    #         print(p, dir)

    rpoints, azimuth = zip(*rpoints)
    (x, y, height) = zip(*rpoints)

    rpoints_cyl = []
    for p in rpoints:
        azi, angle = cartesian_to_cylindrical(cam_xz_t + (cam_pos.get('y'),), p)
        
        rpoints_cyl.append((azi, angle))

    azi, ang = zip(*rpoints_cyl)

    long_line = (azi, ang)

    find_align(long_line, short_line)
    
    camx, camz = cam_xz_t
    #fig, (ax1, ax2) = plt.subplots(2)
    plt.figure(1)
    plt.imshow(raster)
    plt.scatter(x, y, s=10, color='r')
    plt.scatter(camx, camz, s=15, color='g')

    plt.figure(2)
    plt.imshow(img)
    plt.scatter(short_line[0], short_line[1], s=10, color='y')

    plt.figure(3)
    plt.plot(azimuth, ang)
    plt.ylim([-math.pi/4, math.pi/4])
    plt.gca().set_aspect(1)

    plt.show()

def find_align(long_line, short_line):
    print(len(long_line))
    slx, sly = short_line
    llx, lly = long_line
    print(llx[:10], lly[:10])

def transform_coord(x, z, center_x, center_z, extents_x, extents_z, cell_size):
    xt = (x - center_x + extents_x)/cell_size
    zt = (z - center_z + extents_z)/cell_size

    return (xt, zt)

def importJson(fname):
    try:
        with open(fname) as f:
            content = f.read()
            scene_data  = json.loads(content)
            return scene_data

    except IOError:
        print(f"cannot find file with path: {fname}")        

def raster_map_from_flat(raster_flat, x, z):
    arr = np.asarray(raster_flat)
    arr = np.reshape(arr, (z, x))
    return arr


def drawraster(raster):
    print(raster.shape)
    plt.imshow(raster)
    plt.show()

if __name__ == "__main__":
    main()


