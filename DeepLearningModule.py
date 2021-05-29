from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class handDetector():
    def __init__(self):
        self.finger_path1 = 'finger_demo'
        self.categories1 = ['0', '1', '2', '3', '4', '5']
        self.num_class1 = len(self.categories1)
        X = []
        Y = []
        for idx, category in enumerate(self.categories1):
            label = [0 for i in range(self.num_class1)]
            label[idx] = 1
            image_dir = self.finger_path1 + '/' + category + '/'
            for top, dir, f in os.walk(image_dir):
                #        f.remove('.DS_Store')
                for filename in f:
                    img = cv2.imread(image_dir + filename)
                    X.append(img / 128)
                    Y.append(label)

        """
        self.finger_path2 = 'fingers'
        self.categories2 = ['zero', 'one', 'two', 'three', 'four', 'five']
        self.num_class2 = len(self.categories2)
        Xx = []
        Yy = []
        for idx, category in enumerate(self.categories2):
            label = [0 for i in range(self.num_class2)]
            label[idx] = 1
            image_dir = self.finger_path2 + '/' + category + '/'
            for top, dir, f in os.walk(image_dir):
                #        f.remove('.DS_Store')
                for filename in f:
                    img = cv2.imread(image_dir + filename)
                    Xx.append(img / 128)
                    Yy.append(label)
        """

        Xtr = np.array(X)
        Ytr = np.array(Y)
#        Xte = np.array(Xx)
#        Yte = np.array(Yy)
        self.X_train, self.Y_train = Xtr, Ytr
#        self.X_test, self.Y_test = Xte, Yte

        self.model = Sequential()
        self.model.add(Conv2D(16, 3, 3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(20, 3, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=10, validation_split=0.2, epochs=5, verbose=0)
        # print(model.summary())

    def predictHands(self, img):
        self.predeict = self.model.predict(img)
        return img




def main():
    pTime = 0   # previous time
    cTime = 0   # current time

    # 카메라에서 영상 가져오기
    cap = cv2.VideoCapture(0)
    detector = handDetector()


    while True:
        # 한 장의 이미지(frame)을 가져오기
        # 영상 : 이미지(frame)의 연속
        # 정상적으로 읽어왔는지 -> success
        # 읽어온 이미지 -> img
        success, img = cap.read()
        if not(success):    # frame 정보를 정상적으로 읽지 못하면
            break   # while문 빠져나가기

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        cTime = time.time()     # current time
        fps = 1 / (cTime - pTime)   # frame rate : 디스플레이 장치가 화면 하나의 데이터를 표시하는 속도
        pTime = cTime

        ## 웹캠으로 입력받은 이미지에 frame rate 출력 ##
        #cv2.putText(img,    # 웹캠으로 입력받은 이미지에
        #            str(int(fps)),  # frame rate 를 출력
        #            (10, 70),   # 출력될 위치 설정
        #            cv2.FONT_HERSHEY_PLAIN, # 사용할 font
        #            3,  # font scale
        #            (255, 0, 255),  # 색 : 보라색
        #            3)   # 굵기


        # 이미지 보여주기
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # frameRate = 1 동안 한 프레임을 보여준다


# 이 모듈을 실행하면 main 실행
if __name__ == "__main__":
    main()