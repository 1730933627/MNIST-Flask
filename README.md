# MNIST手写数字识别Web应用

>
> 在这个手写数字识别web应用中，使用了Python来编写整个项目，使用tensorflow和keras来建立模型，将建立的模型保存到根目录下，利用base64接收web发送来的图像数据，利用opencv2来处理图像数据，再用建立的模型读取处理完的数据，发送会web应用，已完成对数字的预测。Web端利用canvas画出图像，利用Jquery发送canvas的图像和接收Flask返回的数据，并将呈现在页面上。
>

## 安装需要模块
需要的模块都在requirements.txt,请自行安装。
## 构造并保存该模型
运行model.py
## 开启flask
运行app.py