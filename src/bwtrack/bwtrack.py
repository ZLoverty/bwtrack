import numpy as np
from myimagelib.myImageLib import to8bit, matlab_style_gauss2D, bestcolor
from scipy.signal import convolve2d
from myimagelib.xcorr_funcs import normxcorr2, FastPeakFind
import pandas as pd
from skimage.feature import peak_local_max
from skimage import io, measure, draw

def find_black(img, size=7, thres=None, std_thres=None, plot_hist=False):
    """
    Find black particles in 乔哥's image. 
    
    :param img: raw image to find particles in 
    :type img: 2d array
    :param size: diameter of particle (px)
    :type size: int
    :param thres: threshold of mean intensity for discerning black and white particles. If None, the function will plot a histogram of mean intensity to help us.
    :type thres: int
    :param std_thres: threshold of standard deviation for discerning black and white particles. If None, the function will plot a histogram of standard deviation to help us.
    
    .. note::
    
       If ``thres=None`` or ``std_thres=None``, all detected features will be returned. Histograms of mean intensity and standard deviation will be plotted to help us set the threshold.

    :return: list of particle positions, pixel value sums and corr map peak values (x, y, pv, peak)
    :rtype: pandas.DataFrame
    
    .. rubric:: Edit
    
    * Nov 16, 2022: Initial commit.
    * Dec 09, 2022: Speed up by replacing the sum loop with ``regionprops``. Plot histograms to help setting threshold. Include distance check.
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
    
    # apply min_dist criterion
    particles = pd.DataFrame({"x": peak[1], "y": peak[0]})
    # 加入corr map峰值，为后续去重合服务
    particles["peak"] = corr[particles.y, particles.x]
    particles = min_dist_criterion(particles, size)
    
    # 计算mask内的像素值的均值和标准差
    ## Create mask with feature regions as 1
    R = size / 2.
    mask = np.zeros(img.shape)
    for num, i in particles.iterrows():
        rr, cc = draw.disk((i.y, i.x), 0.8*R) # 0.8 to avoid overlap
        mask[rr, cc] = 1
        
    ## generate labeled image and construct regionprops
    label_img = measure.label(mask)
    regions = measure.regionprops_table(label_img, intensity_image=img, properties=("label", "centroid", "intensity_mean", "image_intensity")) # use raw image for computing properties
    table = pd.DataFrame(regions)
    table["stdev"] = table["image_intensity"].map(np.std)
       
    if thres is not None and std_thres is not None:
        table = table.loc[(table["intensity_mean"] <= thres)&(table["stdev"] <= std_thres)]
    elif thres is None and std_thres is None:
        print("Threshold value(s) are missing, all detected features are returned.")        
    elif thres is not None and std_thres is None:
        print("Standard deviation threshold is not set, only apply mean intensity threshold")
        table = table.loc[table["intensity_mean"] <= thres]
    elif thres is None and std_thres is not None:
        print("Mean intensity threshold is not set, only apply standard deviation threshold")
        table = table.loc[table["stdev"] <= std_thres]
    
    if plot_hist == True:
        table.hist(column=["intensity_mean", "stdev"], bins=20)
        
    table = table.rename(columns={"centroid-0": "y", "centroid-1": "x"}).drop(columns=["image_intensity"])
    
    return table


def find_white(img, size=7, mask_size=None, mask_pattern="dw", thres=None, std_thres=None, plot_hist=False):
    """
    Find "white" particles in Qiaoge's images of particles at water-oil interface.
    
    :param img: input image
    :param size: particle size to look for (px)
    :param mask_size: mask (template) image size (px). By default, ``mask_size`` is set to ``size+2``. Making mask slightly larger can help making the correlation map sharper, sometimes. 
    :param mask_pattern: choose from "mh", "gs" and "dw", which stands for mexican hat, gaussian and double well, respectively. 
    :param thres: mean intensity threshold, meant to discern white particles from black particles.
    :param std_thres: standard deviation threshold, meant to discern white particles from black particles.
    :plot_hist: if True, plot mean intensity and standard deviation histogram. This can help you to determine good threshold values. Set to False when doing batch tracking (default).
    :return particles: a list of particle coordinates detected.

    .. rubric:: Edit
    
    * Jan 20, 2023 -- add double well mask pattern.
    """
    if mask_size == None:
        mask_size = size + 2 # set default mask_size value 
     
    # construct mask
    if mask_pattern == "dw":
        X, Y = np.mgrid[-mask_size/2: mask_size/2: mask_size*1j, -mask_size/2: mask_size/2: mask_size*1j]
        tpl = 1/4 * (X**2 + Y**2) ** 2 - size**2/8 * (X**2 + Y**2) 
    elif mask_pattern == "mh":
        tpl = mexican_hat(shape=(mask_size, mask_size), sigma=0.8) # sigma here is still arbitrary, I don't know how to set a more heuristic value yet
    elif mask_pattern == "gs":
        tpl =matlab_style_gauss2D(shape=(mask_size, mask_size), sigma=mask_size/3)

    img = to8bit(img) # convert to 8-bit and saturate

    corr = normxcorr2(tpl, img, "same")
    coordinates = peak_local_max(corr, min_distance=5) 
    
    # apply min_dist criterion
    particles = pd.DataFrame({"x": coordinates.T[1], "y": coordinates.T[0]})
    # 加入corr map峰值，为后续去重合服务
    particles["peak"] = corr[particles.y, particles.x]
    # particles = min_dist_criterion(particles, size)
    
    # 计算mask内的像素值的均值和标准差
    ## Create mask with feature regions as 1
    R = size / 2.
    mask = np.zeros(img.shape)
    for num, i in particles.iterrows():
        rr, cc = draw.disk((i.y, i.x), 0.8*R) # 0.8 to avoid overlap
        mask[rr, cc] = 1

    ## generate labeled image and construct regionprops
    label_img = measure.label(mask)
    regions = measure.regionprops_table(label_img, intensity_image=img, properties=("label", "centroid", "intensity_mean", "image_intensity")) # use raw image for computing properties
    table = pd.DataFrame(regions)
    table["stdev"] = table["image_intensity"].map(np.std)
    
    ## Arbitrary lower bound here, be careful!
    intensity_lb = (table["intensity_mean"].median() + table["intensity_mean"].mean()) / 4
    table = table.loc[table["intensity_mean"]>=intensity_lb]
    
    if thres is not None and std_thres is not None:
        table = table.loc[(table["intensity_mean"] <= thres)&(table["stdev"] <= std_thres)]
    elif thres is None and std_thres is None:
        print("Threshold value(s) are missing, all detected features are returned.")        
    elif thres is not None and std_thres is None:
        print("Standard deviation threshold is not set, only apply mean intensity threshold")
        table = table.loc[table["intensity_mean"] <= thres]
    elif thres is None and std_thres is not None:
        print("Mean intensity threshold is not set, only apply standard deviation threshold")
        table = table.loc[table["stdev"] <= std_thres]
    
    if plot_hist == True:
        table.hist(column=["intensity_mean", "stdev"], bins=20)
    
    table = table.rename(columns={"centroid-0": "y", "centroid-1": "x"}).drop(columns=["image_intensity"])

    return table

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