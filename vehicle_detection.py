# Referenced from https://imageai.readthedocs.io/en/latest/detection/index.html

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images\\inputimage4.png"), output_image_path=os.path.join(execution_path , "images\\imagenew.png"))
detections = detector.detectObjectsFromImage(input_image="inputimage4.png", output_image_path="imagenew.png")

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

# img = cv2.imread('images\\inputimage4.png', 1)
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# img2 = cv2.imread('images\\imagenew1.png', 1)
# plt.subplot(122),plt.imshow(img2)
# plt.title('New Image'), plt.xticks([]), plt.yticks([])