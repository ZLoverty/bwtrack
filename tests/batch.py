from bwtrack.bwtrack import *
import matplotlib.pyplot as plt
from myimagelib.myImageLib import readdata
import sys, os
from skimage import io

"""
Find black and white particles in a tif sequence. It generates a .csv file containing particle coordinates, as well as a .jpg sequence of images annotated with particle locations.

.. rubric:: Syntax

python batch.py folder

* folder -- the folder containing tif images. 
"""

folder = sys.argv[1]

save_folder = os.path.join(folder, "result")
if os.path.exists(save_folder) == False:
    os.makedirs(save_folder)

l = readdata(folder, "tif")

dpi = 300

black_list = []
white_list = []
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    h, w = img.shape
    black = find_black(img, size=7, thres=100)
    white = find_white(img, size=7)
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(img, cmap="gray")
    draw_particles(black, color="magenta")
    draw_particles(white, color="cyan")
    fig.savefig(os.path.join(save_folder, "{}.jpg".format(i.Name)))
    plt.close()
    black_list.append(black.assign(frame=num))
    white_list.append(white.assign(frame=num))
pd.concat(black_list).to_csv(os.path.join(save_folder, "black.csv"))
pd.concat(white_list).to_csv(os.path.join(save_folder, "white.csv"))