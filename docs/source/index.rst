.. bwtrack documentation master file, created by
   sphinx-quickstart on Fri Jan 20 13:06:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bwtrack's documentation!
===================================

bwtrack is a Python package for tracking black and white particles in the mini interfacial rheometer developed by my friend Qiaoge. See source code and notebooks on `GitHub <https://github.com/ZLoverty/bwtrack>`_.

乔哥牛逼！

I've got very timely and insightful feedbacks from Qiaoge, and I'm optimistic that the method here will become useful for other purposes eventually. So I create this repo to record the progress of the core functions. 

I've always been fascinated about how a software package is made available to users from general public. I have downloaded many Python packages from PyPI. It is part of my habit to check out if there are existing projects that aim to solve my problem, and I'm more than happy to go through the documentations and see if there is a chance to participate in the project.

I've also been interested in publishing the code I write for other people to use. This is the key motivation for me to write code in a maintainable way, with clear comments and documentations. I tried to convince other people to try my code, but every time I have a feeling that I'm confusing other people with some functions in my local library, which do not exist on their computer. The problem is, I import my own functions as if they are from available packages, but they are not. For example, I often write in my code

.. code-block:: python

   from myimagelib.myImageLib import readdata


and this almost always cause problem to other people. Because they will get an error "myImageLib not found". Users with some Python experience will try

.. code-block:: console

   pip install myimagelib --upgrade


only to find there is no package called this. Then they come ask me, and I explain them: you need to go to my GitHub and download this piece of code to your computer, and set this directory as your Python path. Not many steps, but can be confusing at each step, and ... not easy. 

Imagine, if I can make myImageLib a PyPI package, and state it as a requirement of other code I give to users, things can be much more straightforward!

So this code, along with myImageLib, will be the first package I'm going to publish. The goal is to make it downloadable through ``pip``, and work properly right after downloading.

Contents
--------

.. toctree::

   :maxdepth: 1

   usage
   api
   notebooks
