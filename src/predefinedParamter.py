import numpy as np
##########################################
# This is a module for predefined parameter
# WIDTH = 320
# HEIGHT = 240
# KERNEL1
# KERNEL2
# ROI_VERTICES
# CANNY MIN_VAL MAX_VAL
# MIN CONTOURS LEN
# CENTER_SHIFT
# Add more parameters....................
#########################################
image_width = 320
image_height = 240
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
vertices = np.array([[0,240],[0,90],[320,90],[320,240],], np.int32)
canny_min_val = 240
canny_max_val = 250
min_contours_len = 50
center_shift = 10
