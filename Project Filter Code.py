#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib notebook
import numpy as np
import cv2
import imageio
import sys
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageChops
from scipy.signal import gaussian, convolve2d
from scipy import misc, ndimage
from skimage import color, data, restoration, io
from skimage.viewer import ImageViewer


# In[2]:


img_name = 'images/inputimage.png'


# In[3]:


# Using openCV Library

image = cv2.imread(img_name, 0)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('input_image', image)
M, N = image.shape
print(M, N)

height = 7
width = 7

# Mean Filter
new_image1 = cv2.blur(image, (height, width))
cv2.imshow('Mean Filter', new_image1)

# Gaussian Filter
new_image2 = cv2.GaussianBlur(image, (height, width), 0)
cv2.imshow('Gaussian Filter', new_image2)

# Median Filter
new_image3 = cv2.medianBlur(image, height)
cv2.imshow('Median Filter', new_image3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


'''
cv2.fastNlMeansDenoising() - works with a single grayscale images
cv2.fastNlMeansDenoisingColored() - works with a color image.
cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
'''
img = cv2.imread(img_name)
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
dst2 = cv2.fastNlMeansDenoising(img)

plt.figure(0), plt.title("Input Image"), plt.imshow(img), plt.show()
plt.figure(1), plt.title("Image1"), plt.imshow(dst), plt.show()
plt.figure(2), plt.title("Image2"), plt.imshow(dst2), plt.show()


# In[5]:


# Using PIL

im = Image.open(img_name)
im = im.convert('LA')  # Grayscale

# im.save("img1.png")

im1_1 = im.filter(ImageFilter.BLUR)
im1_2 = im.filter(ImageFilter.FIND_EDGES)
im1_3 = im.filter(ImageFilter.EDGE_ENHANCE)
im1_4 = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
im1_5 = im.filter(ImageFilter.EMBOSS)
im1_6 = im.filter(ImageFilter.CONTOUR)
im1_7 = im.filter(ImageFilter.DETAIL)
im1_8 = im.filter(ImageFilter.SHARPEN)
im1_9 = im.filter(ImageFilter.SMOOTH)
im1_10 = im.filter(ImageFilter.SMOOTH_MORE)

Image._show(im1_1, title="ImageFilter.BLUR")
Image._show(im1_2, title="ImageFilter.FIND_EDGES")
Image._show(im1_3, title="ImageFilter.EDGE_ENHANCE")
Image._show(im1_4, title="ImageFilter.EDGE_ENHANCE_MORE")
Image._show(im1_5, title="ImageFilter.EMBOSS")
Image._show(im1_6, title="ImageFilter.CONTOUR")
Image._show(im1_7, title="ImageFilter.DETAIL")
Image._show(im1_8, title="ImageFilter.SHARPEN")
Image._show(im1_9, title="ImageFilter.SMOOTH")
Image._show(im1_10, title="ImageFilter.SMOOTH_MORE")

Image._show(ImageChops.subtract(im,im1_1), title="im-ImageFilter.BLUR")
Image._show(ImageChops.subtract(im,im1_2), title="im-ImageFilter.FIND_EDGES")
Image._show(ImageChops.subtract(im,im1_3), title="im-ImageFilter.EDGE_ENHANCE")
Image._show(ImageChops.subtract(im,im1_4), title="im-ImageFilter.EDGE_ENHANCE_MORE")
Image._show(ImageChops.subtract(im,im1_5), title="im-ImageFilter.EMBOSS")
Image._show(ImageChops.subtract(im,im1_6), title="im-ImageFilter.CONTOUR")
Image._show(ImageChops.subtract(im,im1_7), title="im-ImageFilter.DETAIL")
Image._show(ImageChops.subtract(im,im1_8), title="im-ImageFilter.SHARPEN")
Image._show(ImageChops.subtract(im,im1_9), title="im-ImageFilter.SMOOTH")
Image._show(ImageChops.subtract(im,im1_10), title="im-ImageFilter.SMOOTH_MORE")


# In[6]:


im1 = im.filter(ImageFilter.MinFilter(size=3))
im2 = im.filter(ImageFilter.MaxFilter(size=3))
im3 = im.filter(ImageFilter.MedianFilter(size=3))
im4 = im.filter(ImageFilter.ModeFilter(size=3))
im5 = im.filter(ImageFilter.GaussianBlur(radius=3))
im6 = im.filter(ImageFilter.BoxBlur(radius=3))
im7 = im.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))
im8 = im.filter(ImageFilter.Kernel(size=(3,3), kernel=np.ones(9), scale=1, offset=0))
im9 = im.filter(ImageFilter.RankFilter(size=3, rank=3))

Image._show(im1, title="ImageFilter.MinFilter")
Image._show(im2, title="ImageFilter.MaxFilter")
Image._show(im3, title="ImageFilter.MedianFilter")
Image._show(im4, title="ImageFilter.ModeFilter")
Image._show(im5, title="ImageFilter.GaussianBlur")
Image._show(im6, title="ImageFilter.BoxBlur")
Image._show(im7, title="ImageFilter.UnsharpMask")
Image._show(im8, title="ImageFilter.Kernel")
Image._show(im9, title="ImageFilter.RankFilter")

Image._show(ImageChops.subtract(im,im1), title="im-ImageFilter.MinFilter")
Image._show(ImageChops.subtract(im,im2), title="im-ImageFilter.MaxFilter")
Image._show(ImageChops.subtract(im,im3), title="ImageFilter.MedianFilter")
Image._show(ImageChops.subtract(im,im4), title="im-ImageFilter.ModeFilter")
Image._show(ImageChops.subtract(im,im5), title="im-ImageFilter.GaussianBlur")
Image._show(ImageChops.subtract(im,im6), title="im-ImageFilter.BoxBlur")
Image._show(ImageChops.subtract(im,im7), title="im-ImageFilter.UnsharpMask")
Image._show(ImageChops.subtract(im,im8), title="im-ImageFilter.Kernel")
Image._show(ImageChops.subtract(im,im9), title="im-ImageFilter.RankFilter")


# In[10]:


img = np.float32(io.imread(img_name, as_gray=True))
# img = color.rgb2gray(io.imread('image.jpg'))
img = color.rgb2gray(img)

psf = np.ones((7, 7))/49
# img = convolve2d(img, psf, 'same')

# Add noise
# img += 0.01 * img.std() * np.random.standard_normal(img.shape)

deconv_img = restoration.wiener(img, psf, 1100)
deconv_img2, _ = restoration.unsupervised_wiener(img, psf)

# ImageViewer(img).show()
# ImageViewer(deconv_img).show()
# ImageViewer(deconv_img2).show()

cv2.imshow("Input Image", img)
cv2.imshow("Deconv1 Image", deconv_img)
cv2.imshow("Deconv2 Image", deconv_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


def arithmetic_mean(img, m, n):
    M, N = img.shape
    print(M, N)
    p = (m-1)//2
    q = (n-1)//2
    print(p, q)
    img = np.pad(img, (p,q), mode='constant')
    M, N = img.shape
    print(M, N)
    print("Running...")
    for i in range(p, M - p):
        for j in range(q, N - q):
            val = 0
            for k in range(m):
                for l in range(n):
                    val += img[i - p + k][j - q + l]
            img[i][j] = val/(m*n)
    print("Running complete\n")
    return img

def geometric_mean(img, m, n):
    M, N = img.shape
    print(M, N)
    p = (m-1)//2
    q = (n-1)//2
    print(p, q)
    img = np.pad(img, (p,q), mode='constant', constant_values=255)
    M, N = img.shape
    print(M, N)
    print("Running...")
    for i in range(p, M - p):
        for j in range(q, N - q):
            val = 1
            for k in range(m):
                for l in range(n):
                    if img[i - p + k][j - q + l] != 0:
                        val *= img[i - p + k][j - q + l]
            img[i][j] = val**(1/(m*n))
    print("Running complete\n")
    return img

def harmonic_mean(img, m, n):
    M, N = img.shape
    print(M, N)
    p = (m-1)//2
    q = (n-1)//2
    print(p, q)
    img = np.pad(img, (p,q), mode='constant', constant_values=255)
    M, N = img.shape
    print(M, N)
    print("Running...")
    for i in range(p, M - p):
        for j in range(q, N - q):
            val = 0
            for k in range(m):
                for l in range(n):
                    if img[i - p + k][j - q + l] != 0:
                        val += 1/img[i - p + k][j - q + l]
                    else:
                        val += 1
            img[i][j] = (m*n)/val
    print("Running complete\n")
    return img

def sharp_masking(img):  # using Laplacian filter [0, -1, 0; -1, 4, -1; 0, -1, 0]
    M, N = img.shape
    print(M, N)
    p = q = 1
    img = np.pad(img, (p,q), mode='constant')
    M, N = img.shape
    print(M, N)
    print("Running...")
    for i in range(1, M-1):
        for j in range(1, N-1):
            img[i][j] = (5*img[i][j] - img[i-1][j] - img[i][j-1] - img[i][j+1] - img[i+1][j])%256
    print("Running complete\n")
    return img

def sharp_masking2(img):  # using Laplacian filter [-1, -1, -1; -1, 8, -1; -1, -1, -1]
    M, N = img.shape
    print(M, N)
    p = q = 1
    img = np.pad(img, (p,q), mode='constant')
    M, N = img.shape
    print(M, N)
    print("Running...")
    for i in range(1, M-1):
        for j in range(1, N-1):
            img[i][j] = (8*img[i][j] - img[i-1][j-1] - img[i-1][j] - img[i-1][j+1] - img[i][j-1] - img[i][j+1] - img[i+1][j-1] - img[i+1][j] - img[i+1][j+1])%256
    print("Running complete\n")
    return img

def unsharp_masking(img, m):  # using Box/Blur Filter (1/m^2)*ones(m^2).reshape(m, m)
    M, N = img.shape
    print(M, N)
    p = q = (m-1)//2
    print("Running...")
    filter = m**(-2)
    for i in range(p, M - p):
        for j in range(q, N - q):
            val = 0
            for k in range(m):
                for l in range(m):
                    val += img[i - p + k][j - q + l]
            val = val*filter
            img[i][j] = 2*img[i][j] - val
    print("Running complete\n")
    return img


# In[12]:


# img = np.float32(np.array(io.imread(img_name, as_gray=True)))
# ImageViewer(img).show()
img = cv2.imread(img_name, 0)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

am_img = arithmetic_mean(img, 3, 3)
gm_img = geometric_mean(img, 3, 3)
hm_img = harmonic_mean(img, 3, 3)
sm_img = sharp_masking(img)
sm2_img = sharp_masking2(img)
um_img = unsharp_masking(img, 7)

cv2.imshow("Input Image", np.float32(img))
cv2.imshow("AM Image", np.float32(am_img))
cv2.imshow("GM Image", np.float32(gm_img))
cv2.imshow("HM Image", np.float32(hm_img))
cv2.imshow("Sharp Masking 1 Image", np.float32(sm_img))
cv2.imshow("Sharp Masking 2 Image", np.float32(sm2_img))
cv2.imshow("Unharp Masking Image", np.float32(um_img))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


def _arithmetic_mean(img, m, n):
    M, N = img.shape
    print(M, N)
    filter = np.ones((m, n))*(1/m*n)
    print("Running...")
    img = cv2.filter2D(img, -1, filter)
    print("Running complete\n")
    return img

def _sharp_masking(img):  # using Laplacian filter [0, -1, 0; -1, 4, -1; 0, -1, 0]
    M, N = img.shape
    print(M, N)
    filter = np.array([[0,-1,0], [-1, 4, -1], [0, -1, 0]])
    print("Running...")
    grad2f = cv2.filter2D(img, -1, filter)
    img = img + grad2f
    print("Running complete\n")
    return img

def _sharp_masking2(img):  # using Laplacian filter [-1, -1, -1; -1, 8, -1; -1, -1, -1]
    M, N = img.shape
    print(M, N)
    filter = np.array([[-1,-1,-1], [-1, 8, -1], [-1, -1, -1]])
    print("Running...")
    grad2f = cv2.filter2D(img, -1, filter)
    img = img + grad2f
    print("Running complete\n")
    return img

def _unsharp_masking(img, m):  # using Box/Blur Filter (1/m^2)*ones(m^2).reshape(m, m)
    M, N = img.shape
    print(M, N)
    box_filter = np.ones((m,m))/(m**2)
    print("Running...")
    filtered = cv2.filter2D(img, -1, box_filter)
    mask = img - filtered
    img = img + mask
    print("Running complete\n")
    return img


# In[14]:


# img = np.float32(np.array(io.imread(img_name, as_gray=True)))
# ImageViewer(img).show()
img = cv2.imread(img_name, 0)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

_am_img = _arithmetic_mean(img, 7, 7)
_sm_img = _sharp_masking(img)
_sm2_img = _sharp_masking2(img)
_um_img = _unsharp_masking(img, 7)

cv2.imshow("Input Image",np.float32(img))
cv2.imshow("_AM Image",np.float32(_am_img))
cv2.imshow("_Sharp Masking 1 Image",np.float32(_sm_img))
cv2.imshow("_Sharp Masking 2 Image",np.float32(_sm2_img))
cv2.imshow("_Unsharp Masking 2 Image",np.float32(_um_img))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




