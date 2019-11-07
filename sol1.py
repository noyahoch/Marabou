import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
import heapq
import colorsys

# TODO NORMALIZE


TO_YIQ = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],
          [0.212, -0.523, 0.311]]

MAX_TONE_LEVEL = 255
MIN_TONE_LEVEL = 0


def check_RGB(image):
    if len(image.shape) == 3:
        return True
    return False


'''
Q1 
'''


def read_image(filename, representation):
    image = imread(filename)
    if representation == 1 and check_RGB(image):
        image = rgb2gray(image)
    if representation == 2:
        image = image.astype(np.float64) / 255
    image = np.array(image)
    return image


'''
Q2
'''


def imdisplay(filename, representation):
    image = read_image(filename, representation)
    if representation == 1:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)
    plt.show()


'''
Q3 
 '''


def convert_image(im, trans_mat):
    h = len(im)
    w = len(im[0])
    mat = np.ndarray(shape=(h, w, 3))
    for k in range(3):
        mat[:, :, 0] = trans_mat[0][0] * im[:, :, 0] + trans_mat[0][1] * im[:,
                                                                         :,
                                                                         1] + \
                       trans_mat[0][2] * im[:, :, 2]
        mat[:, :, 1] = trans_mat[1][0] * im[:, :, 0] + trans_mat[1][1] * im[:,
                                                                         :,
                                                                         1] + \
                       trans_mat[1][2] * im[:, :, 2]
        mat[:, :, 2] = trans_mat[2][0] * im[:, :, 0] + trans_mat[2][1] * im[:,
                                                                         :,
                                                                         1] + \
                       trans_mat[2][2] * im[:, :, 2]
    return mat


def rgb2yiq(imRGB):
    yiq = convert_image(imRGB, TO_YIQ)
    return yiq


def yiq2rgb(imYIQ):
    TO_RGB = np.linalg.inv(TO_YIQ)
    rgb = convert_image(imYIQ, TO_RGB)
    return rgb


'''
Q4 what is bin edges , z-1?, round T
TODO noramalize and check bugs 
'''


def histogram_equalize(im_orig):
    y = im_orig
    if check_RGB(im_orig):
        print("rgb")
        yiq = rgb2yiq(im_orig)
        y = yiq[:, :, 0]
    y = y * 255

    # HIST ORIG
    hist_orig, bins_orig = np.histogram(y, range(257))
    # IM_EQ
    eq_hist = np.cumsum(hist_orig)

    N = len(y[0]) * len(y)
    eq_hist = eq_hist / N
    min_val = np.min(eq_hist[np.nonzero(eq_hist)])
    eq_hist = np.around(((eq_hist - min_val) / (eq_hist[255] - min_val))
                        * 255)
    im_eq = eq_hist[y.astype(np.int64)]
    hist_eq, hist_bins = np.histogram(im_eq, range(257))

    if check_RGB(im_orig):
        yiq[:, :, 0] = im_eq / 255
        im_eq = yiq2rgb(yiq)

    plt.imshow(im_orig, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(im_eq, cmap=plt.cm.gray)
    plt.show()

    plt.plot((bins_orig[:-1] + bins_orig[1:]) / 2, hist_orig, color='pink',
             label="Original hist")
    plt.tick_params(labelsize=20)
    plt.plot((hist_bins[:-1] + hist_bins[1:]) / 2, hist_eq, color='green',
             label="Eq hist")

    plt.show()
    first_sum = np.cumsum(hist_orig)
    second_sum = np.cumsum(hist_eq)
    plt.plot((hist_bins[:-1] + hist_bins[1:]) / 2, first_sum, color='green',
             label="Eq hist")
    plt.plot((hist_bins[:-1] + hist_bins[1:]) / 2, second_sum, color='blue',
             label="Eq hist")
    plt.show()
    return [im_eq, hist_orig, hist_eq]


'''
Q5 
'''


def compute_z(n_quant, hist, iter, q):
    # print(np.linspace(MIN_TONE_LEVEL, MAX_TONE_LEVEL,
    # num=n_quant+1)/MAX_TONE_LEVEL) z = np.quantile(cdf, (np.linspace(
    # MIN_TONE_LEVEL, MAX_TONE_LEVEL, num=n_quant+1))/MAX_TONE_LEVEL)
    if iter == 0:
        cdf = np.cumsum(hist)
        pixels = cdf[-1]
        split = np.linspace(0, pixels, num=n_quant+1)
        z = np.searchsorted(cdf, split)
    else:
        z = (q[:-1] + q[1:])/2
        z = np.insert(z, [0,len(z)], [MIN_TONE_LEVEL,MAX_TONE_LEVEL])
    print("this is z after iter ", iter , " : ", z)
    return z.astype(np.int64)


def compute_q(hist, z):
    q = np.zeros(len(z)-1)
    for i in range(len(z)-1):
        weights = hist[z[i]:z[i+1]]
        all_sum = sum(hist[z[i]:z[i+1]])
        rang = np.array([i for i in range(z[i],z[i+1])])
        q[i] = (np.dot(rang, weights)/all_sum)
    return q


def quantize(im_orig, n_quant, n_iter):
    y = im_orig
    yiq = []
    if check_RGB(im_orig):
        print("rgb")
        yiq = rgb2yiq(im_orig)
        y = yiq[:, :, 0]

    y = y *255
    z = []
    q = []
    for i in range(n_iter):
        hist, bin = np.histogram(y, 256, (0, 255))
        last_z = z
        z = compute_z(n_quant, hist, i, q)
        last_q = q
        q = compute_q(hist, z)
        if (np.array_equal(last_z,z) or np.array_equal(last_q,q)):
            print("break")
            break
    all_y = np.searchsorted(z,y, side='left')
    im_quant = q[all_y.astype(np.int64)-1]

    if check_RGB(im_orig):
        yiq[:, :, 0] = im_quant / 255
        im_quant = yiq2rgb(yiq)
    plt.imshow(im_quant, cmap='gray')
    plt.title("after"+ str(n_iter))
    plt.show()


if __name__ == '__main__':
    image = read_image('girl.png', 1)

    quantize(image, 8,5)
    # histogram_equalize(image)
