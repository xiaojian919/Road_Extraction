from flask import Flask, request, send_file, Response, make_response
from PIL import Image
import io
import base64
import cv2
import numpy as np
from test_model.Detect2 import *
from test_model.Detect2 import predict_whole_picture

app = Flask(__name__)  # 确保Flask调用的是当前模块
basedir = os.path.abspath(os.path.dirname(__file__))


@app.route("/", methods=["post"])
def demo():
    # 服务端获取客户端发送的图片，并且展示出来
    # str = request.form.get("name")
    file = request.files.get("filename")

    img_byte = file.read()
    # print(img_byte)  # 字节流
    # print(type(img_byte))  # <class 'bytes'>

    image = Image.open(io.BytesIO(img_byte))  # 图片二进制数据

    abs_path = predict_whole_picture(image, step=256)  # 输出图片的绝对路径
    print(abs_path)

    # print(np.shape(img_out))
    # print(type(img_out))

    # return send_file(io.BytesIO(res), mimetype='image/png',
    #                  as_attachment=True, attachment_filename='result.jpg'
    #                  )

    image_data = open(abs_path, "rb").read()
    response = make_response(image_data)
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == '__main__':
    # app.run()
    app.run(host="192.168.1.8", port=51, debug=True)



