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
    fname = 'skyline.jpg'
    # fname = 'unitycity.jpg'
    # fname = 'sky2.jpg'
    # fname='skylab.jpg'
    # fname='screenshot.png'
    bgr = cv.imread(fname, cv.IMREAD_COLOR)
    res = close_open(5, 10, bgr)
    canny_on_blue_channel(bgr)
    # rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    # b, g, r = cv.split(bgr)
    # gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23, 2)
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    # closing = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)
    
    # contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # canny = cv.Canny(gray, 64, 26)

    # longest = max(contours, key=lambda x: cv.arcLength(x, True))

    # fig, axs = plt.subplots(2,3)

    # axs[0,0].imshow(rgb)
    # axs[0,1].imshow(gray,'gray')
    # axs[0,2].imshow(canny,'gray')
    # axs[1,0].imshow(th,'gray')
    
def close_open(r1, r2, bgr):


def canny_on_blue_channel(bgr):
    blue, g, t = cv.split(bgr)
    canny = cv.Canny(blue, 64, 26)
    fig, axs = plt.subplots(2,3)

    axs[0,0].imshow(bgr)
    axs[0,1].imshow(canny,'gray')
    axs[0,2].imshow(canny,'gray')
    axs[1,0].imshow(canny,'gray')
    
    plt.show()

if __name__ == "__main__":
    main()