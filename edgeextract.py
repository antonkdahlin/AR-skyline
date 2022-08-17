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
    # src = '5 5 1 2 4;5 5 1 3 2;2 4 2 5 3;3 3 2 0 2;5 1 5 3 0'
    # src = np.matrix(src).astype('uint8')
    print(f'opencv version {cv.__version__}')
    

    
    
    extract_skyline_with_preprocessing('skyline.jpg', True)

def extract_skyline_with_preprocessing(fname, plot = False):
    bgr = cv.imread(fname, cv.IMREAD_COLOR)
    blue, g, r = cv.split(bgr)
    morph = close_open(9, 9, blue)
    
    morph_close = close_open(10, 1, blue)
    morph_open = close_open(1, 10, blue)

    canny = canny_on_blue_channel(morph)
    ath = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 8)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


    res = extract_skyline(canny)
    
    
    if plot:
        fig, axs = plt.subplots(3,3)
        fig2, axs2 = plt.subplots(1,3)
        def imshowhelp(axs, data, txt, xlabel):
            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_xlabel(xlabel)
            axs.title.set_text(txt)
            axs.imshow(data, 'gray')

        fig2.set_size_inches(4,8)
        
        imshowhelp(axs2[0], blue, 'source', '(a)')
        imshowhelp(axs2[1], morph_close, 'closing 10', '(b)')
        imshowhelp(axs2[2], morph_open, 'opening 10', '(c)')

        axs[0,0].imshow(rgb)    
        axs[0,1].imshow(blue,'gray')
        axs[0,2].imshow(morph,'gray')
        axs[1,0].imshow(canny,'gray')
        axs[1,1].imshow(ath,'gray')
        axs[1,2].imshow(g,'gray')
        axs[2,0].imshow(r,'gray')


        
        plt.tight_layout()
        # fig2.savefig('morphological_operations_demo.jpg', bbox_inches='tight', dpi = 400)
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

def getStructuringElement_circle(d):
    kernel = np.zeros((d,d), dtype='uint8')
    xl = np.linspace(-1,1,d)
    yl = np.linspace(-1,1,d)
    xx,yy = np.meshgrid(xl,yl)
    # y,x = np.ogrid[-r:r+1,-r:r+1]
    # mask = x**2 + y**2 <= r**2
    mask = xx**2 + yy**2 <= 1
    kernel[mask] = 1
    return kernel

def close_open(r1, r2, img):
    
    # kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r1, r1))
    # kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r2, r2))
    kernel_close = getStructuringElement_circle(r1)
    kernel_open = getStructuringElement_circle(r2)
    #print(kernel_close)
    # print(kernel_open)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_close)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_open)
    return opening

def canny_on_blue_channel(blue):
    canny = cv.Canny(blue, 32, 64)
    return canny

if __name__ == "__main__":
    main()