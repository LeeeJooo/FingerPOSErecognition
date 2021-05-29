from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os
import cv2
import numpy as np

# Train Dataset 만들기
finger_path = 'finger_demo'
#categories = ['zero']
categories = ['0', '1', '2', '3', '4', '5']
num_class = len(categories)
print(num_class)
X = []
Y = []
for idx, category in enumerate(categories):
    label = [0 for i in range(num_class)]
    label[idx] = 1
#    print(label)
#    image_dir = finger_path
    image_dir = finger_path + '/' + category + '/'

#    print(image_dir)
    for top, dir, f in os.walk(image_dir):
        # os.walk 는 하위 폴더들을 for문으로 탐색할 수 있게 해줌
        # 인자로 전달된 path에 대해서 다음 3개의 값이 있는 tuple을 넘겨줌
        # root: dir과 files가 있는 path
        # dirs: root 아래에 있는 폴더들
        # files: root 아래에 있는 파일들
#        f.remove('.DS_Store')
#        print("root : ", top)
#        print("dir : ", dir)
#        print("files : ", f)
        for filename in f:
            img = cv2.imread(image_dir + filename)
            # img : filename Image의 객체 행렬을 return 받음
#            print(img.shape)    # (128, 128, 3)
            # 이미지는 3차원 행렬로 return
            # 128은 행(Y축), 128은 열(X축), 3은 행과 열이 만나는 지점의 값이 몇개의 원소로 이루어져 있는지를 나타냄
            # 위 값의 의미는 이미지의 사이즈가 128 * 128 이라는 의미
            # 3은 색을 표현하는 BGR값을 의미

            X.append(img/128)
            Y.append(label)

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
            img = cv2.imread(image_dir + filename)
            Xx.append(img/128)
            Yy.append(label)

#print(X)
#print(Y)

Xtr = np.array(X)
Ytr = np.array(Y)
Xte = np.array(Xx)
Yte = np.array(Yy)
X_train, Y_train = Xtr, Ytr
X_test, Y_test = Xte, Yte
#print(Xtr)
#print(Ytr)
print(X_train.shape)    # (5, 128, 128, 3), 5는 이미지 갯수
print(Y_train.shape)    # (5, 1), 5는 이미지 갯수

# 합성곱 신경망 구성하기
# 필터로 특징을 뽑아주는 컨볼루션 레이어
# tf.keras.models 모듈의 Sequential 클래스를 사용해서 인공신경망의 각 층을 순서대로 쌓을 수 있다
model = Sequential()
# add() 메서드를 이용해서 합성곱 층 Conv2D와 Max pooling 층 MaxPooling2D를 반복해서 구성
# Conv2D : 영상 처리에 주로 사용
# 첫번째 인자 : 컨볼루션 필터의 수
#   - 필터는 가중치를 의미. 하나의 필터가 입력 이미지를 순회하면서 적용된 결과값을 모으면 출력 이미지가 생성된다.
# 두번째 인자 : 컨볼루션 커널의 (행, 열)
# padding : 경계처리방법을 정의
#   - 'valid' : 유효한 영역만 출력이 된다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작다
#   - 'same' : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일하도록 입력 이미지 경계에 빈 영역(0)을 추가하여 필터를 적용. !!!입력 이미지에 경계를 핛흡시키는 효과가 있음!!!
# input_shape : 샘플 수를 제외한 입력 형태를 정의. 모델에서 첫 레이어일 때만 정의하면 된다.
#   - (행, 열, 채널 수) 로저 정의, 흑백영상인 경우 채널이 1이고, 컬러(RGB/BGR)영상인 경우에는 채널을 3으로 설정
# activation : 활성화 함수 설정.
#   - linear : 디폴트값, 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나온다
#   - relu : rectifier 함수, 은닉층에 주로 사용
#   - sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 사용
#   - softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 사용
# 출력형태
#   - image_data_format 이 'channels_first'인 경우 : (샘플 수, 필터 수, 행, 열) 로 이루어진 4D 텐서
#   - image_data_format 이 'channels_last'인 경우 : (샘플 수, 행, 열, 필터 수) 로 이루어진 4D 텐서
#   - 행과 열의 크기는 padding = 'same' 인 경우에는 입력 형태의 행과 열의 크기가 동일
model.add(Conv2D(16,    # 첫번째 Conv2D 층의 첫번째 인자는 filters 값이다
                        # 합성곱 연산에서 사용되는 filter는 이미지에서 특징을 분리해내는 기능을 한다
                        # filters의 값은 합성곱에 사용되는 필터의 종류(개수)이며, 출력 공간의 차원(깊이)를 결정한
                 3, 3,  # 두번째 인자는 kernel_size 이다
                        # kernel_size 는 합성곱에 사용되는 필터(=커널)의 크기 이다
                        # (3*3) 크기의 필터가 사용되면
                 padding='same',    # same 옵션: padding이 존재하여 입력과 출력의 크기는 같다
                 activation='relu',     # 활성화 함수(Activation function)은 'relu'로 지정
                 input_shape=X_train.shape[1:]))    # 입력 데이터의 형태는 (128, 128, 3)으로 설정

# Pooling은 합성곱에 의해 얻어진 Feature map 으로부터 값을 샘플링해서 정보를 압축하는 과정을 의미
# Max Pooling 레이어: 사소한 변화를 무시해줌
#   - 사소한 변화를 무시해줌
#   - 컨볼루션 레이어의 출력 이미지에서 주요한 값만 뽑아 크기가 작은 출력 영상을 만든다. 이것은 지역적인 사소한 변화가 영향을 미치지 않도록 한다
#   - 주요 인자: pool_size
#       * 수직, 수평 축소 비율을 지정한다
#       * (2, 2) 이면 출력 영상 크기는 입력 영상 크기의 반으로 줄어든다
model.add(MaxPooling2D(pool_size=(2,2)))    # Max_Pooling은 특정 영역에서 가장 큰 값을 샘플링하는 풀링 방식
                                            # 풀링 필터의 크기를 2*2 영역으로 설정
model.add(Dropout(0.25))

model.add(Convolution2D(20,
                        3,3,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
"""
model.add(Convolution2D(64,
                        3, 3,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,
                        3, 3,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,
                        3, 3,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
"""

# Flatten 레이어
#   - 영상을 일차원으로 바꿔준다
#   - CNN에서 컨볼루션 레이어나 맥스풀링 레이어를 반복적으로 거치면 주요 특징만 추출되고, 추출된 주요 특징은 전결합층에 전달되어 학습된다.
#   - 컨볼루션 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다루지만, 전결합층에 전달하기 위해서는 1차원 자료로 바꿔주어야 한다.
#   - 이전 레이어의 출력 정보를 이용하여 입력 정보를 자동으로 설정하며, 출력 형태는 입력 형태에 따라 자동으로 계산된다 -> 파라미터를 지정해줄 필요가 없다
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())

# 모델 학습과정 설정하기
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# compile() 메서드의 파라미터
# optimizer : 훈련 과정을 설정, Adam, SGD 등이 있다
# loss : 최적화 과정에서 최소화될 손실함수를 설정. MSE(평균 제곱 오차)와 binary_crossentropy가 자주 사용된다
# metrics : 훈련을 모니터링하기 위해 사용된다.

# 모델 학습시키기
#model.fit(X_train, Y_train, epochs=1)
history = model.fit(X_train, Y_train, batch_size =10, validation_split = 0.2, epochs = 5, verbose = 0)
#print(model.summary())

# 모델 학습과정 살펴보기
print('## training loss and acc ##')
print(history.history['loss'])
print(history.history['val_loss'])
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.legend(['training', 'validation'], loc = 'upper left')
#plt.show()
# history 객체: fit 함수의 반환 값
# loss : 매 에포크 마다의 훈련 손실값
# acc : 매 에포크 마다의 훈련 정확도
# val_loss : 매 에포크 마다의 검증 손실값
# val_acc : 매 에포크 마다의 검증 정확도

# 모델 평가하기
results = model.evaluate(X_test, Y_test, batch_size=32)
print('Test accuracy: ', str(results))


# 모델 사용하기
Y_predict = model.predict(X_test)
print(Y_predict.shape)
print(len(Y_test))
correct = 0
for i in range(len(Y_test)):
    print(" True : ", np.argmax(Y_test[i]), ", Predict : ", np.argmax(Y_predict[i]))
    if (np.argmax(Y_test[i]) == np.argmax(Y_predict[i])) :
        correct += 1
print(len(Y_test), " 중 ", correct, " 개 일치")

# 모델 저장하기
#model.save('hand_detect_model.h5')