from math import tan, asinh, degrees, pi, radians, cos, log, modf
import itertools
import functools

import numpy as np
from matplotlib import image

from matplotlib import pyplot as plt

import os

from PIL import Image

import requests

def meters_per_pixel(lat_deg, zoom):
    # assuming earth circumference at equator = 40 075 016.686 meters
    # assuming each tile is 256x256 in size
    # assuming earth is a perfect sphere
    return 40075016.686*cos(radians(lat_deg))/2**(zoom+8)


def latlon_to_world(lat_deg, lon_deg):
    # possible place to optimize math, numbers cancel out and bit shifting is possible
    lat_rad = radians(lat_deg)
    x = (lon_deg + 180)*256/360
    y = (pi - log(tan(pi/4+lat_rad/2)))*256/(2*pi)
    return x, max(min(y,256),0)



def world_to_pixel(wx, wy, zoom):
    return wx*2**zoom, wy*2**zoom

def pixel_to_tile(pxx, pxy):
    xf, xi = modf(pxx/256)
    yf, yi = modf(pxy/256)
    return (xi, xf),(yi, yf)


def get_tile(lat_deg, lon_deg, zoom):
    # https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
    lat_rad = radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - asinh(tan(lat_rad)) / pi) / 2.0 * n
    return int(xtile), int(ytile)

def get_tiles(min_lat_deg, max_lat_deg, min_lon_deg, max_lon_deg, zoom):
    # handle cases when lon jumps from 180 to -180 etc..
    xmin, ymax = get_tile(min_lat_deg, min_lon_deg, zoom)
    xmax, ymin = get_tile(max_lat_deg, max_lon_deg, zoom)
    res = []
    
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            res.append((x, y))
    return res  

class Point:
    def __init__(self, lat_deg, lon_deg):
        self.lat = lat_deg
        self.lon = lon_deg
    
    def world(self):
        lat_rad = radians(self.lat)
        x = (self.lon + 180)*256/360
        y = (pi - log(tan(pi/4+lat_rad/2)))*256/(2*pi)
        return x, max(min(y,256),0)
    
    def pixel(self, zoom):
        wx, wy = self.world()
        return wx*2**zoom, wy*2**zoom

    def tile(self, zoom):
        pxx, pxy = self.pixel(zoom)
        xf, xi = modf(pxx/256)
        yf, yi = modf(pxy/256)
        return (xi, xf),(yi, yf)

def download_tiles(tiles, zoom):
    url_normal = 'https://s3.amazonaws.com/elevation-tiles-prod/normal/{z}/{x}/{y}.png'
    url_terrarium = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
    
    for x, y in tiles:
        fname = f'fuji-{x}-{y}-{zoom}.png'
        if not os.path.isfile(f'./{fname}'):
            response = requests.get(url_terrarium.format(z=zoom, x=x, y=y))
            
            with open(fname, "wb") as f:
                f.write(response.content)

'''map is a raster height map containg data from one or multiple tiles. 
The map is centered around arguments lat_deg and lon_deg.
Map lat interval [lat_deg - radius, lat_deg + radius]
Map lon interval [lon_deg - radius, lat_don + radius]
zoom will alter the resolution of the map'''
def create_map(lat_deg, lon_deg, zoom, radius): 
    sw_point = Point(lat_deg - radius, lon_deg - radius) # sw
    ne_point = Point(lat_deg + radius, lon_deg + radius) # ne

    tiles = get_tiles(lat_deg - radius, 
                      lat_deg + radius,
                      lon_deg - radius,
                      lon_deg + radius,
                      zoom)

    download_tiles(tiles, zoom)

    # calculating cuttoff points
    sw_tile = sw_point.tile(zoom)
    (min_tile_x, swtilepartx),(max_tile_y, swtileparty) = sw_tile

    left_cutoff = int(swtilepartx * 256)
    bottom_cutoff = int(swtileparty * 256)

    ne_tile = ne_point.tile(zoom)
    (max_tile_x, netilepartx),(min_tile_y, netileparty) = ne_tile

    right_cutoff = int(netilepartx * 256)
    top_cutoff = int(netileparty * 256)

    num_tiles_x = max_tile_x - min_tile_x + 1
    num_tiles_y = max_tile_y - min_tile_y + 1
    #print(left_cutoff,bottom_cutoff)
   
    map = []
    for rowi in range(int(min_tile_y), int(max_tile_y) + 1):
        x_range = range(int(min_tile_x), int(max_tile_x) + 1)
        row = [image.imread(f'fuji-{coli}-{rowi}-{zoom}.png') for coli in x_range]
        row = np.concatenate(row, axis = 1)

        map.append(row)

    map = np.concatenate(map, axis = 0)
    map = map[
        top_cutoff:bottom_cutoff+256*int(num_tiles_y-1), 
        left_cutoff:right_cutoff+256*int(num_tiles_x-1)]

    (sw_x_pixel, _) = sw_point.pixel(zoom)
    (_, ne_y_pixel) = ne_point.pixel(zoom)
    res = {'pixel_offset_x': sw_x_pixel, 'pixel_offset_y': ne_y_pixel, 'map': map, 'mpp': meters_per_pixel(lat_deg, zoom)}

    return res

    # plt.imshow(map)
    # plt.show()

   
# class Map:
#     def __init__(self, lat_deg, lon_deg, zoom, radius) -> None:
#         self.center_lat_deg = lat_deg
#         self.center_lon_deg = lon_deg
#         self.radius = radius
#         self.zoom = zoom



    
    


def main():
    
    url_normal = 'https://s3.amazonaws.com/elevation-tiles-prod/normal/{z}/{x}/{y}.png'
    url_terrarium = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
    # lat,lon = 35.363053, 138.730243
    lat, lon = 27.98993013928877, 86.92526718587251
    zoom = 10
    create_map(lat,lon, zoom, 0.3)

    

    # xtile, ytile = get_tile(lat, lon, zoom)

    # response = requests.get(url_terrarium.format(z=zoom, x=xtile, y=ytile))
    # print(xtile, ytile)
    # with open('test.png', "wb") as f:
    #     f.write(response.content)

def test_latlon_to_world():
    assert latlon_to_world(0, 0) == (128.0, 128.0)
    assert latlon_to_world(90, 180) == (256.0, 0)
    


def test_get_tile():
    assert get_tile(35.363053, 138.730243, 10) == (906, 404)
    assert get_tile(35.066855134413665, 138.72170503395913, 10) == (906, 405)

    assert get_tile(35.36122418559648, 138.73163496885547, 11) == (1813, 808) # ne
    assert get_tile(35.361446410691954, 138.66569107184188, 11) == (1812, 808) # nw
    assert get_tile(35.298754705569245, 138.72972750076002, 11) == (1813, 809) # se
    assert get_tile(35.29630829927816, 138.65996866755557, 11) == (1812, 809) # sw
    
def test_get_tiles():
    assert get_tiles(35.29630829927816, 35.361446410691954, 138.65996866755557, 138.73163496885547, 11) == [(1812, 808),(1813, 808),(1812, 809),(1813, 809)]

if __name__ == "__main__":
    
    test_latlon_to_world()
    test_get_tile()
    test_get_tiles()
    main()