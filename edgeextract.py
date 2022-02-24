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
    bgr = cv.imread('skyline.jpg', cv.IMREAD_COLOR)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    print(gray[0,0])
    print(bgr[0,0])

    fig, axs = plt.subplots(3)

    axs[0].imshow(bgr)
    axs[1].imshow(gray, 'gray')
    axs[2].imshow(gray)
    plt.show()
    

if __name__ == "__main__":
    main()