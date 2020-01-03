# https://github.com/Aqsa-K/Car-Number-Plate-Detection-OpenCV-Python/blob/master/CarPlateDetection.py
# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# https://imageai.readthedocs.io/en/latest/detection/index.html
# https://imageai.readthedocs.io/en/latest/detection/index.html


import cv2
import numpy as np
from matplotlib import pyplot


def canny(s):
    image1 = cv2.imread("s",0)

    size = int(5) // 2
    x, y = numpy.mgrid[-5:6, -5:6]
    n = 1 / (2.0 * numpy.pi * 1.4**2)
    g = numpy.exp(-((x**2 + y**2) / (2.0* 1.4**2))) * n

    gaussian = cv2.filter2D(image1, -1, g) # Convolve

    Sobelx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobely = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Intensityx = cv2.filter2D(gaussian, -1, Sobelx) # Convolve
    Intensityy = cv2.filter2D(gaussian, -1, Sobely) # Convolve
    Gradient = numpy.hypot(Intensityx, Intensityy)
    thetaGradient = numpy.arctan2(Intensityy, Intensityx)



    M, N = Gradient.shape
    Z = numpy.zeros((M,N))
    angle = thetaGradient * 180. / numpy.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = Gradient[i, j+1]
                    r = Gradient[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = Gradient[i+1, j-1]
                    r = Gradient[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = Gradient[i+1, j]
                    r = Gradient[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = Gradient[i-1, j-1]
                    r = Gradient[i+1, j+1]

                if (Gradient[i,j] >= q) and (Gradient[i,j] >= r):
                    Z[i,j] = Gradient[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass


    Tl=0.05
    Th=0.09
    Th = Z.max() * Th;
    Tl = Th * Tl;

    M, N = Z.shape
    res = numpy.zeros((M,N), dtype=numpy.int32)

    weak = numpy.int32(25)
    strong = numpy.int32(255)

    M, N = Z.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (Z[i,j] == weak):
                try:
                    if ((Z[i+1, j-1] == strong) or (Z[i+1, j] == strong) or (Z[i+1, j+1] == strong) or (Z[i, j-1] == strong) or (Z[i, j+1] == strong) or (Z[i-1, j-1] == strong) or (Z[i-1, j] == strong) or (Z[i-1, j+1] == strong)):
                        Z[i, j] = strong
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass


    (thresh,abctemp) = cv2.threshold(numpy.float32(Z),127,255,cv2.THRESH_BINARY)
    cv2.imwrite("temp.png",abctemp)


# def right_side(image):
#     y = 0
#     h = 768
#     x = 650
#     w = 300
#
#     image = plate_det(y, h, x, w, image)
#     return image
#
#
# def left_side(image):
#     # Read the image file
#     # image = cv2.imread('temp2.png')
#     # image = cv2.imread('temp2.png')
#     # image = cv2.imread("cartraffic3.png")
#
#     y = 0
#     h = 768
#     x = 300
#     w = 350
#     image = plate_det(y, h, x, w, image)
#     return image
#

number = 0


def plate_det(y, h, x, w, image):
    crop_img = image[y:y + h, x:x + w]
    # cv2.imwrite("cropped.png",crop_img)
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    image = crop_img
    # Resize the image - change width to 500
    # image = imutils.resize(image, width=500)


    # cv2.imshow("Original Image", image)

    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # filter for noise removal
    imagegray = cv2.bilateralFilter(imagegray, 11, 17, 17)


    imageedged = cv2.Canny(imagegray, 80, 220)


    (imagecontour, new) = cv2.findContours(imageedged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imagecontour = sorted(imagecontour, key=cv2.contourArea, reverse=True)[:30]
    plate = None

    count = 0
    global number
    for i in imagecontour:
        abc = cv2.arcLength(i, True)

        approx = cv2.approxPolyDP(i, 0.02 * abc, True)
        if len(approx) == 5 or len(approx) == 4:
            plate = approx
            area = cv2.contourArea(plate)
            if (area < 700 and area > 650):
                cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)
                # path='/home/d4rkvaibhav/Documents/DIP_proj/final/sample/image'+str(number)+".png"
                # cv2.imwrite(path,image)
                # print("saved",number)
                number += 1

    return image





img = np.float32(cv2.imread('inputimage2.png', 0))
img2 = np.float32(cv2.imread('inputimage3.png', 0))

img3 = np.subtract(img, img2)

cv2.imwrite("sub_img.png", img3)

img4 = cv2.imread("sub_img.png", 0)

cv2.imwrite("int_image.png", img4)

img5 = cv2.Canny(img4, 200, 250)

cv2.imwrite("final_image.png", img5)


image1 = cv2.imread("final_image.png", 0)
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
# '''
# img_name = 'images/inputimage2.png'
# img = cv2.imread(img_name, 0)
# print(img.shape)
# edges = cv2.Canny(img, 170, 200)
#
# pyplot.subplot(121), pyplot.imshow(img, cmap='gray')
# pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.subplot(122), pyplot.imshow(edges, cmap = 'gray')
# pyplot.title('Edge Image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.show()
# '''


r=[209,223,240]


image1 = cv2.imread("cartraffic3.png")
height, width = image1.shape[:2]
rows,cols = image1.shape[:2]

image2 = cv2.imread("cartraffic8.png")
height1, width1 = image1.shape[:2]
rows1,cols1 = image1.shape[:2]

k = []
count=0
x_row=0
flag123=0
for i in range(rows):
    if flag123==1:
        break
    for j in range(cols):
        a,b,c=(image1[i,j])

        if(r==[c, b, a]):
            print("found a blue light")
            count=0
            for i1 in range(5):
                for j1 in range(5):
                    a, b, c = (image1[i + i1, j + j1])
                    print(image1[i + i1, j + j1])
                    if(r==[c,b,a]):
                        count=count+1
            if(count==25):

                count1 = 0
                for i1 in range(5):
                    for j1 in range(5):
                        a, b, c = (image2[i + i1, j + j1])
                        print(image2[i + i1, j + j1])
                        if (r == [c, b, a]):
                            count1 = count1 + 1
                if(count!=count1):
                    flag123=1
                    print("Emergency vehicle detected")
                    break
print("No emergency vehicle found")



vidObj = cv2.VideoCapture('video_trim.mp4')
count = 0
success = 1



while success:
    # vidObj object calls read
    # function extract frames
    success, image = vidObj.read()
    image_main = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    # cv2.imshow("input",image)
    # cv2.waitKey(0)
    # Saves the frames with frame-count
    # image=left_side(image_main)
    # image2=right_side(image_main)
    y = 500
    h = 768
    x = 0
    w = 1366
    # crop_img = image[y:y+h, x:]
    # cv2.imshow("image",image)
    count += 1
    image2 = plate_det(y, h, x, w, image_main)
    # cv2.imshow('Video Life2Coding', image)
    cv2.imshow('Video 2', image2)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

# from imageai.Detection import ObjectDetection
# import os
#
# execution_path = os.getcwd()
#
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images\\inputimage4.png"), output_image_path=os.path.join(execution_path , "images\\imagenew.png"))
# detections = detector.detectObjectsFromImage(input_image="images\\inputimage4.png", output_image_path="images\\imagenew.png")
#
# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

