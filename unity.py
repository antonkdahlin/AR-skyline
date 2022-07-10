import json
from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from ridge import ridge2, cartesian_to_cylindrical
from edgeextract import extract_skyline_with_preprocessing
import math
from matplotlib.widgets import Slider



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
    # encoded fov below
    encodedhfov = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637896078768153833.json'
    complexscene = 'D:/projects/AR_skyline/unity/ar_skyline/Assets/scenedata/Capture_637897761106632406.json'

    res = importJson(complexscene)
    raster_samplerate = 4 

    # get skyline from image analysis
    imgpath = res.get('absoluteImgPath')
    img, short_line = extract_skyline_with_preprocessing(imgpath)
    
    # gather metadata from image
    vfov_img_degrees = res.get('cam_fov', 90.0) 
    print(f'img resolution {img.shape}')
    height_img, width_img, _ = img.shape
    aspectratio_img = width_img / height_img
    hfov_img_radians = 2*np.arctan(np.tan(np.radians(vfov_img_degrees) / 2 ) * aspectratio_img)
    hfov_img_degrees = hfov_img_radians / math.pi * 180
    img_sample_rate = width_img / hfov_img_degrees # samples per azimuth degree
    print(f'img calculations---------\naspect\t{aspectratio_img}\nhfov\t{hfov_img_degrees}\nvfov\t{vfov_img_degrees}\nsamplerate\t{img_sample_rate}')
    short_line_transformed = transform_img_line(short_line, hfov_img_degrees, img.shape)
    azi, ang = short_line_transformed
    num_data_points = raster_samplerate * int(hfov_img_degrees)
    

    azi_img = np.linspace(azi[0], azi[-1], num_data_points)
    ang_img = np.interp(azi_img, azi, ang)

    
    
    
    print(f'line from img ,xy same length?{len(azi_img) == len(ang_img)}, sampling rate = {len(azi_img) / hfov_img_degrees}')


    
    # generate skyline from raster
    # gather metadata about raster from unity json output file
    raster = raster_map_from_flat(res.get('raster_map'), res.get('grid_size_x'), res.get('grid_size_z'))
    #drawraster(raster)
    raster = np.flip(raster, 0)
    print(res.get("camera_worldPosition").get('x'))
    cam_pos = res.get("camera_worldPosition")
    ground_extents = res.get("ground_extents")
    ground_center = res.get("ground_center")
    cell_size = res.get("raster_cellsize")
    print(f'campos:{cam_pos}')

    # transform coamera from worldspace to raster space
    cam_xz_t = transform_coord(cam_pos.get('x'), cam_pos.get('z'),
        ground_center.get('x'), ground_center.get('z'),
        ground_extents.get('x'), ground_extents.get('z'),
        cell_size)
    campos_xzy = cam_xz_t + (cam_pos.get('y'),)

    print(f'campos raster space:{campos_xzy}')
    
    rpoints = ridge2(raster, cam_xz_t, cam_pos.get('y'), 1, 2*math.pi/(360*raster_samplerate), 0, math.pi*2)
    rpoints_xzy, _  = zip(*rpoints)
    rpoints_x, rpoints_z, _ = zip(*rpoints_xzy)
    azi_rp, ang_rp = handle_ridge_points(campos_xzy, rpoints)
    


    #shift = find_align(long_line, short_line)
    res = np.correlate(ang_rp + ang_rp, ang_img)
    res = sum_of_diff_slide(ang_rp + ang_rp, ang_img)
    idx = np.argmax(res)
    idx = np.argmin(res)
    shift = azi_rp[idx]

    print(f'shift {shift}, dir = {(shift/math.pi*180 + hfov_img_degrees/2 + 90)%360}')
    
    #plot rastermap
    camx, camz = cam_xz_t
    rasterfig, ax = plt.subplots()
    #plt.figure(1)
    ax.imshow(raster)
    rpoints_scat = ax.scatter(rpoints_x, rpoints_z, s=5, color='r')
    camera_pixel = plt.scatter(camx, camz, s=15, color='g')
    ax.set_title("Top down raster-map")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Z")

    # plot image with skyline pixels
    plt.figure(2)
    plt.imshow(img)
    plt.scatter(short_line[0], short_line[1], s=10, color='y')
    plt.title("Skyline pixels ovelayed on image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Z")

    fig, correlation_plot = plt.subplots()
    correlation_plot.plot(res)
    
    fig, line_plot = plt.subplots()
    img_line_plot, = line_plot.plot(azi_img + shift - azi_img[0], ang_img)
    rpline, = line_plot.plot(azi_rp, ang_rp)
    line_plot.set_ylim([-math.pi/4, math.pi/4])
    plt.gca().set_aspect(1)
    line_plot.set_title("Skyline generated from raster-map")
    line_plot.set_xlabel("azimuth radians")
    line_plot.set_ylabel("altitude radians")

    axcamx = plt.axes([0.25, 0.1, 0.65, 0.03])
    camx_slider = Slider(
        ax=axcamx,
        label='camera x pos',
        valmin=-10,
        valmax=10,
        valinit=0,
    )

    # Make a vertically oriented slider to control the amplitude
    axcamz = plt.axes([0.1, 0.25, 0.0225, 0.63])
    camz_slider = Slider(
        ax=axcamz,
        label="camera z pos",
        valmin=-10,
        valmax=10,
        valinit=0,
        orientation="vertical"
    )

    
    def update(val):
        new_cx, new_cz, new_cy = camx + camx_slider.val, camz + camz_slider.val, cam_pos.get('y')
        camera_pixel.set_offsets([new_cx, new_cz])

        rpoints = ridge2(raster, (new_cx, new_cz), new_cy, 1, 2*math.pi/(360*raster_samplerate), 0, math.pi*2)
        rpoints_xzy, _  = zip(*rpoints)
        rpoints_x, rpoints_z, _ = zip(*rpoints_xzy)
        rpoints_scat.set_offsets(list(zip(rpoints_x, rpoints_z)))

        azi_rp, ang_rp = handle_ridge_points((new_cx, new_cz, new_cy), rpoints)
        rpline.set_xdata(azi_rp)
        rpline.set_ydata(ang_rp)

        res = sum_of_diff_slide(ang_rp + ang_rp, ang_img)
        idx = np.argmin(res)
        shift = azi_rp[idx]
        img_line_plot.set_xdata(azi_img + shift - azi_img[0])

        print(f'shift {shift}, dir = {(shift/math.pi*180 + hfov_img_degrees/2 + 90)%360}')

        rasterfig.canvas.draw_idle()
    
    camx_slider.on_changed(update)
    camz_slider.on_changed(update)

    plt.show()

def handle_ridge_points(cam_pos_xzy, rpoints):
    rpoints_cyl = []
    for p, azi  in rpoints:
        raster_pixel_azi, angle = cartesian_to_cylindrical(cam_pos_xzy, p)
        rpoints_cyl.append((azi, angle))

    azi, ang = zip(*rpoints_cyl)
    return azi, ang

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
    '''Transform the skyline from the image which is in pixel-space, xy
    to cylindrical space using the inverse intrinsic matrix.
    Intrinsic matrix is generated from image metadata: aspect and fov.
    This does not account for camera distortion
    '''
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
    
    
    res = np.correlate(lly + lly, sly)
    idx = np.argmax(res)
    shift = llx[idx]
    print(shift)
    
    



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




def intristic_matrix_inv(w, h, hfov_degrees):
    # f=(W/2)/tan(fov/2) https://www.reddit.com/r/computervision/comments/ayclnf/calculate_angle_from_camera_to_detected_object/
    f = w/2/np.tan(np.radians(hfov_degrees)/2)
    
    fx, fy, u0, v0 = f , f , w/2, h/2
    K = np.array([[fx, 0, u0],[0, fy, v0],[0,0,1]])
    Ki = np.linalg.inv(K)

    return Ki

def img_point_to_radians(x, y, Ki):
    '''Transform xy point on image to azimuth and vertical viewing angle using inverse intrinsic matrix of camera
    '''
    r = Ki.dot([x, y, 1])
    ang_radians = np.arctan(r)
    azi = ang_radians.item(0)
    altitude_angle = ang_radians.item(1)
    return azi, altitude_angle


if __name__ == "__main__":

    main()



