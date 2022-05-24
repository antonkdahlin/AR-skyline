import json
from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from ridge import ridge2, cartesian_to_cylindrical
from edgeextract import extract_skyline_with_preprocessing
import math
from scipy import signal


def main():
    low_vfov = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637872826261823552.json'
    vfov90 = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873411250794906.json'
    pos2 = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873477340759380.json'
    test = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873490606180285.json'
    two_cubes = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873505941295330.json'
    fromleft = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873509884645086.json'
    fromright = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873513599114215.json'
    symmetrisk ='D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873538401875630.json'
    two_symmery ='D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637873546905120392.json'
    far_away_two_houses = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637878640973099154.json'
    testing_bb_ofbox = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637878722678594990.json'
    threebuildings = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637883092347188740.json'
    lower_fov = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637885487231920230.json'
    complex_scene = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637885591367803561.json'
    d0 = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637885605140156403.json'
    d50 = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637885606206101013.json'
    test = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637885650953570488.json'

    res = importJson(lower_fov)
    raster_samplerate = 2 

    # get skyline from image analysis
    imgpath = res.get('absoluteImgPath')
    img, short_line = extract_skyline_with_preprocessing(imgpath)
    
    # ----TODO---- get this from camera/unity
    hfov_img_degrees = 90 

    print(f'img resolution {img.shape}')
    height_img, width_img, _ = img.shape
    aspectratio_img = width_img / height_img
    vfov_img_radians = 2*np.arctan(np.tan(np.radians(hfov_img_degrees) / 2 ) / aspectratio_img)
    vfov_img_degrees = vfov_img_radians / math.pi * 180
    img_sample_rate = width_img / hfov_img_degrees # samples per azimuth degree
    print(f'img calculations---------\naspect\t{aspectratio_img}\nhfov\t{hfov_img_degrees}\nvfov\t{vfov_img_degrees}\nsamplerate\t{img_sample_rate}')
    short_line_transformed = transform_img_line(short_line, hfov_img_degrees, img.shape)
    azi, ang = short_line_transformed
    num_data_points = raster_samplerate * hfov_img_degrees
    print()
    azi_new = np.linspace(azi[0], azi[-1], num_data_points)
    ang_new = np.interp(azi_new, azi, ang)

    short_line = azi_new, ang_new
    
    
    print(f'line from img ,xy same length?{len(azi_new) == len(ang_new)}, sampling rate = {len(azi_new) / hfov_img_degrees}')


    
    # generate skyline from raster
    raster = raster_map_from_flat(res.get('raster_map'), res.get('grid_size_x'), res.get('grid_size_z'))
    #drawraster(raster)
    raster = np.flip(raster, 0)
    print(res.get("camera_worldPosition").get('x'))
    cam_pos = res.get("camera_worldPosition")
    ground_extents = res.get("ground_extents")
    ground_center = res.get("ground_center")
    cell_size = res.get("raster_cellsize")
    print(f'campos:{cam_pos}')

    cam_xz_t = transform_coord(cam_pos.get('x'), cam_pos.get('z'),
        ground_center.get('x'), ground_center.get('z'),
        ground_extents.get('x'), ground_extents.get('z'),
        cell_size)
    
    print(f'cam_t:{cam_xz_t}')
    
    
    rpoints = ridge2(raster, cam_xz_t, cam_pos.get('y'), 1, 2*math.pi/(360*raster_samplerate), 0, math.pi*2)
    # print(f'cam_pos: {cam_xz_t + (cam_pos.get("y"),)}')
    
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
    

    #long_line = (azi, ang)
    long_line = (azimuth, ang)
    print(f'line from img ,xy same length?{len(azimuth) == len(ang)}, sampling rate = {len(azimuth) / 360}')

    #shift = find_align(long_line, short_line)
    res = np.correlate(ang + ang, ang_new)
    res = sum_of_diff_slide(ang+ang, ang_new)
    idx = np.argmax(res)
    idx = np.argmin(res)
    shift = azimuth[idx]

    print(f'shift {shift}, dir = {(shift/math.pi*180 + hfov_img_degrees/2 + 90)%360}')
    
    camx, camz = cam_xz_t
    #fig, (ax1, ax2) = plt.subplots(2)
    plt.figure(1)
    plt.imshow(raster)
    plt.scatter(x, y, s=5, color='r')
    plt.scatter(camx, camz, s=15, color='g')
    plt.title("Top down raster-map")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Z")

    plt.figure(2)
    plt.imshow(img)
    plt.scatter(short_line[0], short_line[1], s=10, color='y')
    plt.title("Skyline pixels ovelayed on image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Z")

    
    fig, (line_plot, corr_plot) = plt.subplots(2)
    corr_plot.plot(res)
    line_plot.plot(azi_new + shift + .78, ang_new)
    line_plot.plot(azimuth, ang)
    line_plot.set_ylim([-math.pi/4, math.pi/4])
    plt.gca().set_aspect(1)
    line_plot.set_title("Skyline generated from raster-map")
    line_plot.set_xlabel("azimuth radians")
    line_plot.set_ylabel("altitude radians")
    
    
    plt.show()

def proj_to_cyl(maxval, maxangle):
    def transform(x):
        half_h = maxval/2
        return math.atan(1/half_h*(half_h-x)*math.tan(np.radians(maxangle/2)))

    return  np.vectorize(transform)

# take angle a between 0 and fov
# return between 0 and 1 where angle corresponds on flat image
def angle_to_flat(a, fov):
    # desmos formatting of function y=\sin\left(x\right)\frac{\cos\left(\frac{v}{2}\right)}{\cos\left(x-\frac{v}{2}\right)\cdot\sin\left(v\right)}
    sin=math.sin
    cos=math.cos
    y = sin(a)*cos(fov/2) / (cos(a-fov/2)*sin(fov))
    return y

def sum_of_diff_slide(a, b):
    def sum_of_diff(a, b):
        return sum([abs(a-b) for a,b in zip(a,b)])
    
    return [sum_of_diff(a[x: x+len(b)], b) for x in range(len(a)-len(b)+1)]
        

# sample from a source at provided points
# 0 samples the first element and 1 samples the last element
# linearly interpolates if sample is between two elsements of source
def sample_from(sample_points, source):
    max_index = len(source) - 1
    res = []
    for p in sample_points:
        frac, idx = math.modf(max_index * p)
        if frac == 0.0:
            res.append(source[int(idx)])
        else:
            # case with linear interpolation
            res.append(source[int(idx)] * (1-frac) + source[int(idx) + 1] * frac)
    return res

def transform_img_line(line, img_hfov, img_shape):
    height_img, width_img, _ = img_shape
    Ki = intristic_matrix_inv(width_img, height_img, img_hfov)
    x, y = line[0], line[1]
    
    res = []
    for i in range(width_img):
        azi, ang = img_point_to_radians(x[i], height_img - y[i], Ki)
        res.append((azi, ang))

    azi, ang = zip(*res)
    return (azi, ang)

def find_align(long_line, short_line, hfov_img = 90, ):
    print(f'lenght 0f raster line {len(long_line[0])}, lenght of image: {len(short_line[0])}')
    slx, sly = short_line
    llx, lly = long_line
    
    # slxt = []
    # slyt = []
    # for i in range(len(sly)):
    #     xt, yt = intristic_matrix(slx[i], sly[i])
    #     slxt.append(xt)
    #     slyt.append(yt)

    # plt.figure(7)
    # plt.plot(slxt, slyt)

    # raster_sample_rate = len(llx)/360
    # img_num_samples = hfov_img * raster_sample_rate
    # print(f'sample rate raster: {raster_sample_rate} data points per degree\nnumber of samples from image: {img_num_samples} samples')
    # img_sample_points_degrees = np.linspace(0, hfov_img, int(img_num_samples))
    # img_sample_points_radians = np.radians(img_sample_points_degrees)
    # #print(img_sample_points_radians[:10])
    # x_transformer = lambda a : angle_to_flat(a, np.radians(hfov_img))
    # img_point_on_img = [x_transformer(a) for a in img_sample_points_radians]
    # new_sly = sample_from(img_point_on_img, sly)
    # print(f'lenx {len(img_sample_points_radians)}, leny {len(new_sly)}')

    # # downsample line from image 
    # ratio = 9/16
    # vfov_img = 90
    # hfov_panorama = 360
    # data_points = 361 # ska vara 360, fixa
    # sample_rate_panorama = data_points/hfov_panorama # sample/azimuth degree
    # resample_samplenum = hfov_img * sample_rate_panorama

    # new_sly = sly[::6]
    # new_slx = slx[::6]
    # print(len(sly), len(new_sly))
    # img_height = 1024*ratio
    # #transform = np.vectorize(lambda x: vfov_img*(0.5-x/img_height))
    # transform_y = proj_to_cyl(img_height, vfov_img)
    

    # print(len(new_slx))
    # #sly_transformed = np.radians(transform_f(new_sly))
    # sly_transformed = transform_y(new_sly)
    
    
    res = np.correlate(lly + lly, sly)
    idx = np.argmax(res)
    shift = llx[idx]
    print(shift)
    
    

    

    #plt.show(block=False)

    # print(f'len sly:{len(sly_transformed)}, len lly:{len(lly)}')
    # res = np.correlate(lly + lly, sly_transformed)
    # idx = np.argmax(res)
    
    # print(len(res))

   
    # shift = llx[idx]
    # if shift < 0:
    #     shift = math.pi*2+shift
    
    # print(f'shift {shift}, dir = {(shift/math.pi*180 + hfov_img/2 + 90)%360}')

    # plt.figure(3)
    # plt.plot(np.linspace(shift, np.radians(160) + shift, 171), sly_transformed)
    # plt.plot(np.linspace(0,2*math.pi, len(lly)), res[:len(lly)])


# worldspace to rasterspace
def transform_coord(x, z, center_x, center_z, extents_x, extents_z, cell_size):
    xt = (x - center_x + extents_x)/cell_size
    zt = 2*extents_z-(z - center_z + extents_z)/cell_size

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
    plt.figure('raster')

    plt.gca().invert_yaxis()
    plt.imshow(raster)
    plt.show()

def test_sample_from():
    assert sample_from([0.0,0.5, 1.0],[0,1,2,3,4]) == [0,2,4]

def test_intristic_matrix():
    # res  = []
    # for x in range(0,1024,20):
    #     for y in range(0,576,20):
    #         ray = intristic_matrix(x, y)
    #         azi = ray.item(0)
    #         ang = ray.item(1)
    #         res.append((azi,ang))
    
    # azi, ang = zip(*res)

    # plt.scatter(azi, ang, s=1)
    # plt.show()
    Ki = intristic_matrix_inv(1024, 767, 90)
    for x in range(0, 1024, 20):
        print(img_point_to_radians(x, 400, Ki))

def intristic_matrix_inv(w, h, hfov_degrees):
    # f=(W/2)/tan(fov/2) https://www.reddit.com/r/computervision/comments/ayclnf/calculate_angle_from_camera_to_detected_object/
    f = w/2/np.tan(np.radians(hfov_degrees)/2)
    
    fx, fy, u0, v0 = f , f , w/2, h/2
    K = np.array([[fx, 0, u0],[0, fy, v0],[0,0,1]])
    Ki = np.linalg.inv(K)
    # r = Ki.dot([u,v,1])
    # principal_axis = [0,0,1]

    # print(r)
    # cosangle = r.dot(principal_axis)/(np.linalg.norm(principal_axis)*np.linalg.norm(r))
    # angle_radians = np.arccos(cosangle)
    # print(f'{u},{v},{angle_radians}')
    return Ki

def img_point_to_radians(x, y, Ki):
    r = Ki.dot([x, y, 1])
    ang_radians = np.arctan(r)
    azi = ang_radians.item(0)
    altitude_angle = ang_radians.item(1)
    return azi, altitude_angle

def test_sum_of_diff_slide():
    print(sum_of_diff_slide([1,2,3,4,5], [1,2]))

if __name__ == "__main__":
    #test_intristic_matrix()
    #test_sample_from()
    test_sum_of_diff_slide()
    main()



