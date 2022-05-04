import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def threshholdingwithslider(img):
    def f(x):
        ret, thresh = cv.threshold(img, x, 255, cv.THRESH_BINARY)
        return thresh

    fig, axs = plt.subplots()
    initt = 127
    thresh = f(initt)
    axs.imshow(thresh, 'gray')

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='Frequency [Hz]',
        valmin=0,
        valmax=255,
        valinit=initt,
    )

    def update(val):
        thresh = f(val)
        axs.imshow(thresh, 'gray')
        fig.canvas.draw_idle()

    freq_slider.on_changed(update)

    plt.show()

def main():
    extract_skyline_with_preprocessing('sky2.jpg', True)

def extract_skyline_with_preprocessing(fname, plot = False):
    bgr = cv.imread(fname, cv.IMREAD_COLOR)
    blue, g, r = cv.split(bgr)
    morph = close_open(5, 10, blue)
    canny = canny_on_blue_channel(morph)
    ath = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 8)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


    res = extract_skyline(canny)
    
    
    if plot:
        fig, axs = plt.subplots(2,3)

        axs[0,0].imshow(rgb)
        axs[0,1].imshow(blue,'gray')
        axs[0,2].imshow(morph,'gray')
        axs[1,0].imshow(canny,'gray')
        axs[1,1].imshow(ath,'gray')

        plt.tight_layout()
        plt.show()
    

    return rgb, res

def extract_skyline(img):
    def first_non_zero(array):
        res = -1
        non_zero = np.flatnonzero(array)
        if len(non_zero) > 0:
            res = non_zero[0]
        return res
    first_nonzero = np.apply_along_axis(first_non_zero, 0, img)

    
    
    y, x = img.shape
    
    xval = np.arange(x)
    '''
    plt.imshow(img)
    plt.scatter(xval,first_nonzero)
    plt.show()
    '''
    return (xval, first_nonzero)

def extract_skyline_traverse(img):
    i = 0
    fcol_nonzero = np.flatnonzero(img[:, i])
    y, x = img.shape
    while fcol_nonzero.size < 1 and i < x-1:
        i += 1
        fcol_nonzero = np.flatnonzero(img[:, i])
        
    if fcol_nonzero.size < 1:
        raise ValueError('image has no edge pixels')

    s = [(i, fcol_nonzero[0])]
    
    print(s)
    

def close_open(r1, r2, img):
    # close with disk size 5
    # open with disk size 10
    kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel10 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel5)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel10)
    return opening

def canny_on_blue_channel(blue):
    canny = cv.Canny(blue, 32, 64)
    return canny

if __name__ == "__main__":
    main()