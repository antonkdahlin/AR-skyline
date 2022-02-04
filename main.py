from random import randrange
import numpy as np
from matplotlib import image

from matplotlib import pyplot as plt

from matplotlib.widgets import Slider

import math



def bresenham(start, end):
    x1,y1 = start
    x2,y2 = end
    
    if abs(x2 - x1) < abs(y2 - y1): 
        # highline, one pixel on each y val because y is steeper than x
        if y2 > y1:
            return [(b,a) for (a,b) in lowline(y1,x1, y2,x2)]
        return [(b,a) for (a,b) in lowline(y2,x2, y1,x1)[::-1]]
    
    # lowline, one pixel on each x val because x is steeper than y
    if x2 > x1:
        return lowline(x1,y1, x2,y2)
    return lowline(x2,y2, x1,y1)[::-1]


def lowline(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # increasing or decreasing y
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    di = 2*dy - dx
    y = y1
    res = []
    for x in range(x1, x2+1):
        res.append((x,y))
        if di > 0:
            y = y + yi
            di = di - 2 * dx
        di = di + 2 * dy

    return res

def highline(x1, y1, x2, y2):
    print("doing highline")
    dx = x2 - x1
    dy = y2 - y1

    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    di = 2*dx - dy
    x = x1
    res = []
    for y in range(y1, y2+1):
        res.append((y,x))
        #Z[x,y] = 1
        if di > 0:
            x = x + xi
            di = di - 2 * dy
        di = di + 2 * dx
    return res

def bresenham_demo_display():
    width, height = 32, 32
    fig, ax = plt.subplots()

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    x_slider = Slider(
        ax=axfreq,
        label='X',
        valstep=1.0,
        valmin=0,
        valmax=width-1,
        valinit=width//2,
    )

    # Make a vertically oriented slider to control the amplitude
    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    y_slider = Slider(
        ax=axamp,
        label="Y",
        valstep=1.0,
        valmin=0,
        valmax=height-1,
        valinit=height//2,
        orientation="vertical"
    )

    def update(val):
        Z = np.zeros((height, width))
        line = bresenham((5, 10), (int(x_slider.val), int(y_slider.val)))
        print(line)
        for coord in line:
            Z[coord] = 1
        ax.imshow(Z)
        fig.canvas.draw_idle()

    x_slider.on_changed(update)
    y_slider.on_changed(update)

    plt.show()

# height is how high above the first point the viewpoint is
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




def first_max_index(lst):
    maxi, maxh = (0, lst[0])
    for i, h in enumerate(lst):
        if h > maxh:
            maxi, maxh = i, h
    return maxi

def plot_height(lst):
    z = np.zeros((256, len(lst)))
    for i, ((_y, _x), height) in enumerate(lst):
        z[height, i] = 1

    
    plt.imshow(z[::-1])
    plt.show()

def ridgeline(viewpoint, heightmap, azimuthstart, azimuthend):
    print(viewpoint)
    (x,y,z) = viewpoint
    print(heightmap[(y,x,0)])
    print(heightmap.shape)
    line = bresenham((x,y),(541,226))
    #cp = np.copy(heightmap)
    radius = 200
    #cp[(y,x )] = 255
    res = []
    for azimuth in range(azimuthstart, azimuthend): # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!azimuth span
        target = (x + int(math.cos(math.radians(azimuth)) * radius), y + int(math.sin(math.radians(azimuth)) * radius))
        bline = bresenham((x, y), target)
        hline = [heightmap[(y,x,0)] for (x,y) in bline]
        (rpi, rpa) = ridgepoint(2, hline)
        rpx, rpy = bline[rpi]
        rpz = heightmap[(rpy, rpx, 0)]
        #cp[(rpy, rpx)] = 255
        res.append({'rpx':rpx, 'rpy':rpy, 'rpz':rpz, 'rpa':rpa, 'azimuth':azimuth})
    
    return res

def plot_ridgeline_above_overlaying_heightmap(line, heightmap):
    cp = np.copy(heightmap)
    for point in line:
        cp[(point['rpy'], point['rpx'])] = 255
    plt.imshow(cp)
    plt.show()

def plot_ridgeline_curve(line, azimuthstart, azimuthend):
    # make data
    
    hfov = azimuthend - azimuthstart
    vfov = 90
    vfov_slope = math.tan(math.radians(vfov/2))
    x = np.arange(azimuthstart,azimuthend) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!azimuth span
    y = [math.degrees(math.atan(p['rpa'])) for p in line]
    

    # plot
    fig, ax = plt.subplots(figsize=(4*hfov/vfov,4)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!azimuth span

    ax.plot(x, y, linewidth=1.0)

    ax.set(xlim=(azimuthstart,azimuthend), ylim=(-15, 30)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!azimuth span
    fig.tight_layout()
    plt.show()

    pass


def main():
    # https://tangrams.github.io/heightmapper/#11.65/35.3397/138.7742
    #_image = image.imread("mt_fuji480x239.bmp")
    _image = image.imread("render_noae.bmp")
    data = np.asarray(_image)

    azstart = -90         
    azend = 90

    line = ridgeline((416,232, data[(232, 416, 0)]+2), data, azimuthstart=azstart,azimuthend=azend)
    #plot_ridgeline_above_overlaying_heightmap(line, data)
    plot_ridgeline_curve(line,azimuthstart=azstart,azimuthend=azend)
    # print(data.shape)
    
    # height, width, channels = data.shape
    # viewpoint = (416,232) #x,y
    
    # result = []
    

    # def f(angle):
    #     r = 200
    #     return (416 + int( math.sin( angle ) * r), 232 + int(math.cos(angle) * r))

    # # z:x scale factor 0.040343535335275335 
    # # max ele 3720
    # # min ele 0
    # precision = 360
    # start, end = 0, 2* math.pi
    # step = (end - start) / precision
    # for i in range(precision):
    #     end = f(i * step)
    #     line = bresenham(viewpoint, end)
    #     mapped_line = [data[a, b, 0] for (a,b) in line]
    #     i = first_max_index(mapped_line)
    #     result.append(((line[i]), mapped_line[i]))
    
    # # result = []
    # # for x in range(width):
    # #     line = bresenham(viewpoint, (x, 0))
    # #     mapped_line = [data[a, b, 0] for (a,b) in line]
    # #     i = first_max_index(mapped_line)
    # #     result.append(((line[i]), mapped_line[i]))

    # plot_height(result)


if __name__ == "__main__":
    #bresenham_demo_display()
    main()