# +

from skimage.draw import ellipse_perimeter
import numpy as np



# -

a = 1
b = 11.532562594670797
y0 = np.float64(543.0)
x0 = np.float64(119.5)
orientation = 1.5707963267948966
shape = (544, 446)



max(a, 2)

# %%time
rr, cc = ellipse_perimeter(
    int(round(y0)), int(round(x0)),
    int(round(b)), int(round(a)),
    orientation=-orientation,  
    shape=shape
)

rr

cc
