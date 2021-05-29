from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import os

# 얼굴 검출을 위한 Haar-Cascade 트레이닝 데이터를 읽어 CascadeClassifier 객체를 생성
cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))  # 사람 얼굴 정면에 대한 Haar-Cascade 학습 데이터

# 손가락을 식별하는 모델 호출
model = load_model('hand_detect_model2.h5')

# VideoCapture 객체 생성
cap = cv.VideoCapture(0)

# 라이브로 들어오는 비디오를 frame 별로 캡쳐하고 이를 화면에 display
while True:

    # 재생되는 비디오의 한 frame씩 읽기
    ret, img = cap.read()
    # 비디오 프레임을 제대로 읽었다면 ret 값이 True가 되고 실패하면 False
    if ret == False:
        break

    img_result = img.copy()
    # 이미지 흑백처리
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 히스토그램 평활화(Histogram Equalization)를 적용하여 이미지의 콘트라스트를 향상시킴
    gray = cv.equalizeHist(gray)

    # 얼굴 위치를 리스트로 리턴 (x, y, w, h) / (x, y ):얼굴의 좌상단 위치, (w, h): 가로 세로 크기
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)

    # 얼굴 영역에 검정 사각형 만들기
    height, width = img.shape[:2]
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1 - 10, 0), (x1+x2+10, height), (0, 0, 0), -1)

    # bgr -> hsv 로 변환
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Skin HSV 범위 지정
    low = (0, 30, 0)
    high = (15, 255, 255)

    # 이미지를 binary 이미지로 전환
    img_binary = cv.inRange(img_hsv, low, high)

    # 경계선 찾기
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)

    # binary 이미지에서 윤곽선을 검색
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_contour = None
    max_area = -1

    # 영익이 가장 큰 윤곽선을 선택 : 손 검출
    for contour in contours:
        area = cv.contourArea(contour)
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

    # 검출된 윤곽선을 그린다
    cv.drawContours(img_result, [max_contour], 0, (255, 0, 0), 3)

    # 손 영역의 위치 값을 찾는다
    contours_xy = np.array(max_contour)
    # x의 min과 max 찾기
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)

    # frame에서 손 영역만 자른다
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    img_trim = img[y_min-10:y_max+10, x_min-10:x_max+10]

    # 손 영역 이미지에서 손가락 검출 모델을 이용하여 손가락 모양을 예측한다
    try:
        hand_input = cv.resize(img_trim, (128, 128))
        hand_input = np.expand_dims(hand_input, axis=0)
        hand_input = np.array(hand_input)
        cv.imshow("Result", img_result)
        predictions = model.predict(hand_input)
        print("predict : ", np.argmax(predictions)) # frame에서 손 영역에 윤곽선을 그린 이미지를 반환
        #cv.imshow("Result", img_trim)  # frame에서 손 영역을 자른 이미지를 반환
        cv.waitKey(100)
    except:
        print("손을 인식하지 못했습니다.")
        continue