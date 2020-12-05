# from helpers import gaussian_filter
import cv2
import numpy as np

def gaussian_filter(shape =(15, 15), sigma=1):
    x, y = shape[0] // 2, shape[1] // 2
    gaussian_grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
    gaussian_filter = np.exp(-gaussian_grid)/(2*np.pi*sigma**2)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

image = cv2.imread('cat.jpg')

image = np.float32(image / 255.0)
# cv2.destroyAllWindows()


gfilter = gaussian_filter(sigma=7)
gimage = np.zeros_like(image, dtype=np.float32)
gimage = cv2.filter2D(image, -1, kernel=gfilter)

print(gimage.dtype)
cv2.imshow('a', gimage)
cv2.waitKey(3000)