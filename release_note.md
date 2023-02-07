### Release note

##### v1.4 (2023-02-08)

1. Modify `find_black`
    (i) use smaller kernel for smoothing, 
    (ii) use ``skimage.feature.peak_local_max`` to find peak, 
    (iii) since ``peak_local_max`` has distance criterion already, remove the ``min_dist_criterion`` step. 
    (iv) set gaussian template with larger sigma (less intensity variation).

2. Add `draw_particles`
    - since `show_result` requires inputing image and is not very intuitive, I provide `draw_particles` which only draw a bunch of particles on the axes specified. The color and size of particles can be specified by passing keyword arguments.