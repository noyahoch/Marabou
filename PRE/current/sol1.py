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
GRAYSCAL_REP = 1
COLOR_REP = 2


def check_RGB(image):
    if len(image.shape) == 3:
        return True
    return False


'''
Q1 
'''


def read_image(filename, representation):
    image = imread(filename)
    if representation == GRAYSCAL_REP and check_RGB(image):
        image = rgb2gray(image)
    if representation == COLOR_REP:
        image = image.astype(np.float64) / MAX_TONE_LEVEL
    image = np.array(image)
    return image


'''
Q2
'''


def imdisplay(filename, representation):
    image = read_image(filename, representation)
    plt.imshow(image, cmap='gray')
    plt.show()


'''
Q3 
 '''


def convert_image(im, trans_mat):
    h = len(im)
    w = len(im[0])
    mat = np.ndarray(shape=(h, w, 3))
    for k in range(3):
        mat[:, :, 0] = trans_mat[0][0] * im[:, :, 0] + trans_mat[0][1] * im[:,:,1] + trans_mat[0][2] * im[:, :, 2]
        mat[:, :, 1] = trans_mat[1][0] * im[:, :, 0] + trans_mat[1][1] * im[:,:,1] + trans_mat[1][2] * im[:, :, 2]
        mat[:, :, 2] = trans_mat[2][0] * im[:, :, 0] + trans_mat[2][1] * im[:,:,1] + trans_mat[2][2] * im[:, :, 2]
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
    '''

    :param im_orig:
    :return:
    '''
    y = im_orig * MAX_TONE_LEVEL
    if check_RGB(im_orig):
        yiq = rgb2yiq(im_orig)
        y = yiq[:, :, 0] * MAX_TONE_LEVEL
    hist_orig, bins_orig = np.histogram(y, 256,(0,255))
    eq_hist = np.cumsum(hist_orig)
    N = eq_hist[-1]
    eq_hist = eq_hist / N
    min_val = np.min(eq_hist[np.nonzero(eq_hist)])
    dif = eq_hist[MAX_TONE_LEVEL] - min_val
    eq_hist = np.round(((eq_hist - min_val) / dif) * MAX_TONE_LEVEL)

    im_eq = eq_hist[y.astype(np.int64)]
    hist_eq, hist_bins = np.histogram(im_eq, 256,(0,255))

    if check_RGB(im_orig):
        yiq[:, :, 0] = im_eq / MAX_TONE_LEVEL
        im_eq = yiq2rgb(yiq)

    # plt images - todo delete
    plt.imshow(im_orig, cmap='gray')
    plt.show()
    plt.imshow(im_eq, cmap='gray')
    plt.show()

    plt.plot(hist_orig, color='pink', label="Original hist")
    plt.plot(hist_eq, color='green', label="Eq hist")

    plt.show()
    first_sum = np.cumsum(hist_orig)
    second_sum = np.cumsum(hist_eq)
    plt.plot(first_sum, color='green',
             label="Eq hist")
    plt.plot(second_sum, color='blue',
             label="Eq hist")
    plt.show()
    return [im_eq, hist_orig, hist_eq]


'''
Q5 
'''


def compute_z(q,z):
    '''

    :param q:
    :param z:
    :return:
    '''
    z[1:-1] = np.round((q[:-1] + q[1:])/2)

    return np.round(z)


def compute_q(image, z):
    hist, bins = np.histogram(image,range(257),(0,255))
    q = np.zeros(len(z)-1)
    print(q)
    for i in range(len(z)-1):
        weights = np.array((hist[z[i]:z[i+1]+1]))
        all_sum = sum(hist[z[i]:z[i+1]+1])
        rang = np.array([i for i in range(z[i],z[i+1]+1)])
        q[i] = (np.dot(rang, weights)/all_sum)
    return np.round(q)

def compute_error (image ,q,z):
    hist, bins = np.histogram(image, range(257), (0,255))
    map =  np.searchsorted(z, bins, side = 'left')
    map = np.maximum(map-1,0)[:-1]
    error = np.dot(hist,(bins[:-1]-q[map])**2)
    return error



def quantize(im_orig, n_quant, n_iter):
    y = im_orig
    if check_RGB(im_orig):
        yiq = rgb2yiq(im_orig)
        y = yiq[:, :, 0]

    y = y * 255
    hist,bins = np.histogram(y,range(256), (0,255))
    q = []
# first z#
    cdf = np.cumsum(hist)
    pixels = cdf[-1]
    split = np.linspace(0, pixels, num=n_quant+1)
    z = np.searchsorted(cdf, split)
    z[-1] = 255
    error = np.zeros(n_iter)
    for i in range(n_iter):
        last_z = z.copy()
        q = compute_q(y, z)
        print(q, "q")
        z = compute_z(q,z)
        print(z)
        error.put(i, compute_error(y,q, z))
        if np.array_equal(last_z,z):
            if n_iter -1 > i:
                error = error [:(i+1)]
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

    plt.plot(error, color='pink',label="errors")
    plt.show()

    return [im_quant, error]
if __name__ == '__main__':
    image = read_image('jerusalem.jpg', 2)

    quantize(image, 16,60)
    # print(np.load("C:\\Users\\noyah\\Downloads\\histogram_hist_orig.npy"))
    # histogram_equalize(image)
