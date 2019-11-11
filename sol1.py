import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
from sklearn import cluster

TO_YIQ = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],
          [0.212, -0.523, 0.311]]  # constants that transporm rgb rep to yiq rep

MAX_TONE_LEVEL = 255
MIN_TONE_LEVEL = 0
GRAYSCAL_REP = 1
COLOR_REP = 2
RGB_DIMS = 3

REPRESENTATION_ERROR = "There is not such representation"
QUANTIZE_ERROR = "Iteration or quants are not big enough"
SEGMENT_ERROR = "There is at least one gray-level with more pixels that pixels/n_quants"


def check_RGB(image):
    '''
    checks if an image is from rgb type
    :param image: image as matrix
    :return: True if it's an rgb image, False otherwise
    '''
    if len(image.shape) == RGB_DIMS:
        return True
    return False


'''
Q1 
'''


def read_image(filename, representation):
    '''
    reads an image by its representation.
    :param filename: path of the image
    :param representation: 1 for grayscale, 2 for rgb
    :return: matrix represents the image
    '''
    image = imread(filename)
    if representation == GRAYSCAL_REP:
        if check_RGB(image):
            image = rgb2gray(image)
    elif representation == COLOR_REP:
        image = image.astype(np.float64) / MAX_TONE_LEVEL
    else:
        raise Exception(REPRESENTATION_ERROR)
    image = np.array(image)
    return image


'''
Q2
'''


def imdisplay(filename, representation):
    '''
    displays an image by its' representation
    :param filename: path of the image
    :param representation: 1 for grayscale, 2 for rgb
    '''
    image = read_image(filename, representation)
    plt.imshow(image, cmap='gray')
    plt.show()


'''
Q3 
 '''


def convert_image(im, trans_mat):
    '''
    converts image representation by transformation matrix
    :param im: image
    :param trans_mat
    :return: image converted
    '''
    h = len(im)
    w = len(im[0])
    mat = np.ndarray(shape=(h, w, 3))
    for k in range(3):
        mat[:, :, 0] = trans_mat[0][0] * im[:, :, 0] + \
                       trans_mat[0][1] * im[:, :, 1] + trans_mat[0][2] * im[:, :, 2]
        mat[:, :, 1] = trans_mat[1][0] * im[:, :, 0] + \
                       trans_mat[1][1] * im[:, :, 1] + trans_mat[1][2] * im[:, :, 2]
        mat[:, :, 2] = trans_mat[2][0] * im[:, :, 0] + \
                       trans_mat[2][1] * im[:, :, 1] + trans_mat[2][2] * im[:, :, 2]
    return mat


def rgb2yiq(imRGB):
    '''
    transforms rgb rep to yiq rep
    :param imRGB: image rgb rep
    :return: image yiq rep
    '''
    yiq = convert_image(imRGB, TO_YIQ)
    return yiq


def yiq2rgb(imYIQ):
    '''
    transforms rgb rep to yiq rep
    :param imYIQ: image yiq rep
    :return: image rgb rep
    '''
    TO_RGB = np.linalg.inv(TO_YIQ)
    rgb = convert_image(imYIQ, TO_RGB)
    return rgb


'''
Q4 
'''


def histogram_equalize_helper(y):
    '''

    :param y: grayscale 2d array rep of image
    :return: image after histogram, histogram of the original image, histogram of equalized image
    '''
    y = np.around(y * MAX_TONE_LEVEL)
    y = y.astype(np.int)

    hist_orig, bins_orig = np.histogram(y, MAX_TONE_LEVEL + 1, (0, 255))
    eq_hist = np.cumsum(hist_orig)
    N = eq_hist[-1]
    # eq_hist = eq_hist / N
    min_val = np.min(eq_hist[np.nonzero(eq_hist)])
    dif = eq_hist[MAX_TONE_LEVEL] - min_val
    eq_hist = np.around((eq_hist - min_val) / dif * MAX_TONE_LEVEL)

    im_eq = (eq_hist[y]).astype(np.float64) / MAX_TONE_LEVEL
    hist_eq, hist_bins = np.histogram(im_eq, MAX_TONE_LEVEL + 1, (0, 1))

    return im_eq, hist_orig, hist_eq


def histogram_equalize(im_orig):
    '''
    :param im_orig: image rep
    :return: image after histogram, histogram of the original image, histogram of equalized image
    '''

    if check_RGB(im_orig):
        yiq = rgb2yiq(im_orig)
        im_eq_rgb, hist_orig, hist_eq = histogram_equalize_helper(yiq[:, :, 0])
        yiq[:, :, 0] = im_eq_rgb
        yiq = yiq2rgb(yiq)
        im_eq = np.clip(yiq, 0, 1)
    else:
        im_eq, hist_orig, hist_eq = histogram_equalize_helper(im_orig)
    return [im_eq, hist_orig, hist_eq]


'''
Q5 
'''


def compute_first_z(hist, n_quant):
    '''
    initialize first z array s.t. in every segment will be an equal number of pixels.

    :param hist: image histogram
    :param n_quant: quants number to search by
    :return: z array of segments
    '''
    cdf = np.cumsum(hist)
    if not (hist[np.where(hist >= (cdf[-1] / n_quant))]).shape[0] == 0:
        raise Exception(SEGMENT_ERROR)
    z = np.searchsorted(cdf, np.linspace(0, cdf[-1], num=n_quant + 1))
    z[-1] = MAX_TONE_LEVEL
    return z


def compute_z(q, n_quants):
    '''
    compute segments to quantize
    :param q: mapped values of the segments
    :param z: previous z
    :return: new segments splitting
    '''
    new_z =  np.zeros(n_quants + 1)
    new_z[-1] = MAX_TONE_LEVEL
    new_z = new_z.astype(np.uint8)
    for i in range(1, len(new_z) - 1):
        new_z[i] = np.ceil((q[i - 1] + q[i]) / 2)
    return new_z


def compute_q(hist, z):
    '''
    compute levels to map the segments
    :param    image: original image
    :param z: segments splitting
    :return: new map
    '''
    q = np.zeros(len(z) - 1)
    for i in range(len(z) - 1):
        weighted_sum = np.dot(np.arange(256)[z[i]:z[i + 1]], hist[z[i]:z[i + 1]])
        all_sum = np.sum(hist[z[i]:z[i + 1]])
        q[i] = (weighted_sum / all_sum)
    return q


def compute_error(hist, q, z):
    '''
    compute error rate
    :param image: original image
    :param q: map to segments
    :param z: segments
    :return: error rate
    '''
    nums = np.arange(256)
    error = 0
    for i in range(len(z) - 1):
        error += np.inner(hist[nums[z[i]:z[i + 1]]], ((nums[z[i]:z[i + 1]] - q[i]) ** 2))
    return error


def quantize_helper(y, n_quant, n_iter):
    y = np.around(y * MAX_TONE_LEVEL)
    y = y.astype(np.int)
    hist, bins = np.histogram(y, range(256), (0, 1))
    unique = np.count_nonzero(hist)
    if unique <= n_quant:
        return [y * MAX_TONE_LEVEL, np.array(0)]  # the image already quantize

    z = compute_first_z(hist, n_quant)
    error = []
    q = []
    for i in range(n_iter):
        q = compute_q(hist, z)
        z_next = compute_z(q, n_quant)
        if np.array_equal(z, z_next):
            break
        z = z_next
        error.append(compute_error(hist, q, z))
    y = np.searchsorted(np.round(z), y, side='left').astype(int)
    im_quant = (q[y - 1]).astype(np.float64) / MAX_TONE_LEVEL

    return im_quant, error


def quantize(im_orig, n_quant, n_iter):
    '''
    performs quantization to n_quants by n_iter
    :param im_orig: image
    :param n_quant: segments
    :param n_iter: iterations
    :return: image after quantization, error in each iteration
    '''

    if n_quant <= 0 or n_iter <= 0:
        raise Exception(QUANTIZE_ERROR)

    if check_RGB(im_orig):
        yiq = rgb2yiq(im_orig)
        im_quant, error = quantize_helper(yiq[:, :, 0], n_quant, n_iter)
        yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq)
    else:
        im_quant, error = quantize_helper(im_orig, n_quant, n_iter)

    return [im_quant, error]


'''
BONUS
'''


def quantize_rgb(im_orig, n_quant):
    '''
    quantize rgb image to n_quant colors
    :param im_orig: image
    :param n_quant: colors
    :return: image after quantization
    '''
    width, height, depth = im_orig.shape
    reshaped_im = np.reshape(im_orig, (width * height, depth))
    model = cluster.KMeans(n_clusters=n_quant)
    labels = model.fit_predict(reshaped_im)
    palette = model.cluster_centers_

    quantized_im = np.reshape(palette[labels], (width, height, palette.shape[1]))
    plt.imshow(quantized_im)
    plt.show()
    return quantized_im

