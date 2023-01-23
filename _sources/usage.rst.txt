Usage
=====

Installation
------------

To use bwtrack, first install it using pip:

.. code-block:: console
   
   pip install bwtrack --upgrade

.. note::

   Since this package is under active development, it is a good practice to always look for newer versions by pass ``--upgrade``.


Minimal example
---------------

First, import bwtrack and skimage.

.. code-block:: Python

   from bwtrack.bwtrack import *
   from skimage import io
   import matplotlib.pyplot as plt

Then, load a sample image in python.

.. code-block:: Python
   
   img = io.imread("https://github.com/ZLoverty/bwtrack/raw/main/test_images/large.tif")
   plt.imshow(img, cmap="gray")

.. image:: _static/img.png

Use ``find_white()`` function to locate white particles.

.. code-block:: Python

   particles = find_white(img, size=6, mask_size=7)

Lastly, check the result by plotting detected particles on the raw image.

.. code-block:: Python

   fig, ax = show_result(img, particles, size=6)

.. image:: _static/show-result.png

See **API reference** for more details. 