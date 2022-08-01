import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# 超参数的常数配置
BATCH_SIZE = 100
EPOCHS = 10
VALIDATION_FRACTION_SIZE = 0.10
ROTATION_RANGE = 10
ZOOM_RANGE = 0.10
SHIFT_RANGE = 0.10
SHEAR_RANGE = 0.10


def normalize_x(train: np.ndarray, test: np.ndarray):
    train = train / 255
    test = test / 255
    # mean = train.mean().astype(np.float32)
    # stddev = train.std().astype(np.float32)
    # train = (train - mean) / stddev
    # test = (test - mean) / stddev
    return train, test


# 定义一个比较大的模型
def larger_model():
    # 创建模型
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 加载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 重塑为 [samples][width][height][channels]   //[样本] [宽度] [高度] [通道]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# 将输入从 0-255 标准化为 0-1
X_train, X_test = normalize_x(X_train, X_test)
# 一个热编码输出
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

datagen = ImageDataGenerator(
    rotation_range=ROTATION_RANGE,
    zoom_range=ZOOM_RANGE,
    width_shift_range=SHIFT_RANGE,
    height_shift_range=SHIFT_RANGE,
    shear_range=SHEAR_RANGE,
    validation_split=VALIDATION_FRACTION_SIZE,
)

# 构造模型
model = larger_model()
# 拟合模型
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
model.fit(
    datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        subset='training'
    ),
    validation_data=datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        subset='validation'
    ),
    steps_per_epoch=(X_train.shape[0] * (1 - VALIDATION_FRACTION_SIZE)) // BATCH_SIZE,
    epochs=EPOCHS,
)
# 保存模型
model.save('classifier.h5')

# 模型的最终评估
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))


if __name__ == "__main__":
    print()
