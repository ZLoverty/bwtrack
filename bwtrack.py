import numpy as np
from myImageLib import to8bit, matlab_style_gauss2D
from scipy.signal import convolve2d
from xcorr_funcs import normxcorr2, FastPeakFind
import pandas as pd
from skimage.feature import peak_local_max

def find_black(img, size=7, thres=None):
    """
    Find black particles in 乔哥's image. 
    
    :param img: raw image to find particles in 
    :type img: 2d array
    :param size: diameter of particle (px)
    :type size: int
    :param thres: threshold for discerning black and white particles. By default, ``thres`` is inferred from the data as mean of the median and the mean of pixel sums of all the particles. However, it is preferred to set a threshold manually, with the knowledge of the specific data.
    :type thres: int
    :return: list of particle positions, pixel value sums and corr map peak values (x, y, pv, peak)
    :rtype: pandas.DataFrame
    
    .. rubric:: Edit
    
    :Nov 16, 2022: Initial commit.
    """
    img = to8bit(img) # convert to 8-bit and saturate
    inv_img = 255 - img
    
    # generate gaussian template according to particle size
    gauss_shape = (size, size)
    gauss_sigma = size / 2.   
    gauss_mask = matlab_style_gauss2D(shape=gauss_shape,sigma=gauss_sigma) # 这里的shape是particle的直径，单位px
    
    timg = convolve2d(inv_img, gauss_mask, mode="same") 
    corr = normxcorr2(gauss_mask, timg, "same") # 找匹配mask的位置
    peak = FastPeakFind(corr) # 无差别找峰
    
    # 计算mask内的像素值之和
    Y, X = np.mgrid[0:img.shape[0], 0:img.shape[1]] 
    R = size / 2.
    pixel_sum_list = []
    for y, x in peak.T:
        mask = (X - x) ** 2 + (Y - y) ** 2 < R ** 2
        pv = img[mask].sum()
        pixel_sum_list.append(pv)
        
    # 把数据装进一个DataFrame
    particles = pd.DataFrame({"x": peak[1], "y": peak[0], "pv": pixel_sum_list})
    # 加入corr map峰值，为后续去重合服务
    particles = particles.assign(peak=corr[particles.y, particles.x])
    
    if thres == None:
        thres = (particles.pv.median() + particles.pv.mean()) / 2
        
    return particles.loc[particles.pv <= thres]


def find_white(img, size=7, thres=None):
    
    img = to8bit(img) # convert to 8-bit and saturate
    
    mh = mexican_hat(shape=(5,5), sigma=0.8) # 这里shape和上面同理，sigma需要自行尝试一下，1左右
    #plt.imshow(mh, cmap="gray")

    corr = normxcorr2(mh, img, "same")
    coordinates = peak_local_max(corr, min_distance=5) 
    
    ## Rule out black ones
    Y, X = np.mgrid[0:img.shape[0], 0:img.shape[1]] 
    R = 3.5 
    pixel_sum_new_list = []
    for y, x in coordinates:
        mask = (X - x) ** 2 + (Y - y) ** 2 < R ** 2
        pv = img[mask].sum()
        pixel_sum_new_list.append(pv)

    particles = pd.DataFrame({"x": coordinates.T[1], "y": coordinates.T[0], "pv": pixel_sum_new_list})
    
    if thres == None:
        thres = (particles.pv.median() + particles.pv.mean()) / 4
        
    return particles.loc[particles.pv >= thres]

def min_dist_criterion(coords, min_dist):
    """
    Use minimal distance criterion on a particle coordinate data. 
    
    :param coords: the coordinate data of particles, contains at least two columns (x, y). Optionally, a column (peak) can be included, as the order of the screening.
    :type coords: pandas.DataFrame
    :min_dist: minimal distance allowed between two detected particles.
    :type min_dist: int
    :return: screened coordinates, a subset of coords
    :rtype: pandas.DataFrame
    
    .. rubric:: Edit
    
    :Nov 16, 2022: Initial commit.    
    """
    xy = coords.copy() # create a copy of input DataFrame, just to avoid a warning from pandas
    
    if "peak" in xy: # if we have peak data, sort the data according to peak values
        xy.sort_values(by="peak", ascending=False, inplace=True)
    
    index_to_remove = []
    
    for num, i in xy.iterrows():
        if num not in index_to_remove: # already removed particle should not be considered again
            dist = ((xy.x - i.x) ** 2 + (xy.y - i.y) ** 2) ** 0.5 # distance between particle i and all other particles
            for ind in dist[dist < min_dist].index:
                if ind != num: # exclude itself, because the distance would always be 0, thus < min_dist
                    index_to_remove.append(ind)
    
    return xy.drop(index_to_remove) # drop all the recorded index, and return

def mexican_hat(shape=(3,3), sigma=1):
    """
    2D mexican hat mask
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = 1 / np.pi / sigma ** 4 * (1 - (x*x + y*y) / (2*sigma*sigma)) * np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h