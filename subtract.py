import cv2
import numpy as np
from matplotlib import pyplot

img = np.float32(cv2.imread('inputimage2.png', 0))
img2 = np.float32(cv2.imread('inputimage3.png', 0))

img3 = np.subtract(img, img2)
pyplot.figure(1)
pyplot.subplot(121)
pyplot.imshow(img3, cmap='gray')
pyplot.title('Subtracted Image')
pyplot.xticks([]), pyplot.yticks([])
cv2.imwrite('sub_img.png', img3)

img4 = cv2.imread('sub_img.png', 0)
pyplot.subplot(122)
pyplot.imshow(img4, cmap='gray')
pyplot.title('Int Image')
pyplot.xticks([]), pyplot.yticks([])
cv2.imwrite('int_image.png', img4)

img5 = cv2.Canny(img4, 200, 250)
pyplot.figure(2)
pyplot.subplot(121)
pyplot.imshow(img5, cmap='gray')
pyplot.title('New Image')
pyplot.xticks([]), pyplot.yticks([])
cv2.imwrite('final_image.png', img5)


image1 = cv2.imread('final_image.png', 0)
(thresh, image) = cv2.threshold(np.float32(image1), 127, 255, cv2.THRESH_BINARY)

height, width = image.shape[:2]
print(height)
count = 0
flagprev = 0
flag = 0
for i in range(width):
    flag = 0
    for j in range(100):
        # print(image[height-1-j,i])
        if image[height-1-j, i] == 255:
            flag = i
            break
    if flag != 0:
        break
for i in range(flag):
    for j in range(height):
        image[j, i] = 0
cv2.imwrite("temp.png", image)
print(flag)

height, width = image.shape[:2]
rows, cols = image.shape
k = []
count = 0
x_row = 0
for i in range(rows):
    this_row = []
    for j in range(cols):
        this_row.append((image[height-i-1,j]))
    # print(this_row)
    if 255 in this_row:
        count = x_row
    x_row += 1
print("\nlast white pixel\t", count)
print("totla rows\t\t", x_row)

# Alternate but less generic
'''
img_name = 'images/inputimage2.png'
img = cv2.imread(img_name, 0)
print(img.shape)
edges = cv2.Canny(img, 170, 200)

pyplot.subplot(121), pyplot.imshow(img, cmap='gray')
pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122), pyplot.imshow(edges, cmap = 'gray')
pyplot.title('Edge Image'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()
'''