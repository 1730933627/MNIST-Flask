import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from utils import preprocess, data_uri_to_cv2_img

app = Flask(__name__)
model = load_model("classifier.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 响应web的请求,从 base64 数据 URL 读取图像数据
    imgstring = request.form.get('data')
    
    # 转换为 OpenCV 图像
    img = preprocess(data_uri_to_cv2_img(imgstring))

    data = (img / 255).reshape((1, 28, 28, 1))
    prediction = model.predict(data)
    classes_x = np.argmax(prediction, axis=1)
    predicted_class = classes_x
    # 返回请求
    print("识别为:", predicted_class)
    return f"这个数看起来像: {predicted_class}"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
