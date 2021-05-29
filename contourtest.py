from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import os

cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))
#handnet = cv.dnn.readNet('model/pose_deploy.prototxt', 'model/pose_iter_102000.caffemodel')
model = load_model('hand_detect_model2.h5')

# Test Dataset 만들기
finger_path2 = 'fingers'
categories2 = ['zero', 'one', 'two', 'three', 'four', 'five']
num_class2 = len(categories2)
Xx=[]
Yy=[]
for idx, category in enumerate(categories2):
    label = [0 for i in range(num_class2)]
    label[idx] = 1
    image_dir = finger_path2 + '/' + category + '/'
    for top, dir, f in os.walk(image_dir):
#        f.remove('.DS_Store')
        for filename in f:
            img = cv.imread(image_dir + filename)
            Xx.append(img/128)
            Yy.append(label)
Xte = np.array(Xx)
Yte = np.array(Yy)
X_test, Y_test = Xte, Yte

cap = cv.VideoCapture(0)

while True:

    ret, img = cap.read()

    if ret == False:
        break

    img_bgr = img
    img_result = img_bgr.copy()
    ##img_bgr = removeFaceAra(img_bgr, cascade)
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)  # 이미지 흑백처리
    gray = cv.equalizeHist(gray)
    ##rects = detect(gray, cascade)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)  # 얼굴 위치를 리스트로 리턴 (x, y, w, h) / (x, y ):얼굴의 좌상단 위치, (w, h): 가로 세로 크기


    #print("rects", rects)
    height, width = img_bgr.shape[:2]
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img_bgr, (x1 - 10, 0), (x1+x2+10, height), (0, 0, 0), -1)
    # 얼굴 영역에 검정 사각형 만들기


    ##img_binary = make_mask_image(img_bgr)
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)    # bgr -> hsv 로 변환
    # img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_binary = cv.inRange(img_hsv, low, high)



    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    #cv.imshow("Binary", img_binary)


    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 경계선 그리기
    #for cnt in contours:
    #    cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

    # print("contours : ", contours)
    ## max_area, max_contour = findMaxArea(contours)
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv.contourArea(contour)  # 폐곡선인 contour의 면적

        x, y, w, h = cv.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    cv.drawContours(img_result, [max_contour], 0, (255, 0, 0), 3)

    # print("max_contour : ", max_contour.shape) (1151, 1, 2)
    # print("max_area : ", max_area)

    contours_xy = np.array(max_contour)
    # print(contours_xy.shape)    # ex) (1485, 1, 2)
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
    #print("x_min : ", x_min)
    #print("x_max : ", x_max)

    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
    #print("y_min : ", y_min)
    #print("y_max : ", y_max)

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_trim = imggray[y_min-10:y_max+10, x_min-10:x_max+10]
    #img_trim = cv.cvtColor(img_trim, cv.COLOR_BGR2GRAY)

    #img_trim = img_trim/256

    hand_input = cv.resize(img_trim, (128, 128))
    hand_input = np.expand_dims(hand_input, axis=0)
    hand_input = np.array(hand_input)

    #img_trim = img_trim.resize((128, 128, 3))
    #print("img : ", img_trim.shape) # (488, 255, 3)
    predictions = model.predict(hand_input)

    """
    # test_data 예측
    testPredict = model.predict(X_test)
    print(testPredict.shape)
    print(len(Y_test))
    correct = 0
    for i in range(len(Y_test)):
        print(" TestImage : ", np.argmax(Y_test[i]), ", Predict : ", np.argmax(testPredict[i]))
        if (np.argmax(Y_test[i]) == np.argmax(testPredict[i])):
            correct += 1
    print(len(Y_test), " 중 ", correct, " 개 일치")
    """
    #print("predict : " , modelpredict)
    print(np.argmax(predictions[0]))
    cv.imshow("Result", img_trim)
    #cv.imshow("Result", img_result)
    cv.waitKey(500)