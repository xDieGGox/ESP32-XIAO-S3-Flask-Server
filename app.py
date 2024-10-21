
from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO

import cv2
import numpy as np
import requests

app = Flask(__name__)
_URL = 'http://10.0.0.3'
_PORT = '81'
_ST = '/stream'

#video = cv2.VideoCapture(_URL+':'+_PORT+_ST)
video = cv2.VideoCapture("http://10.0.0.3:81/stream")
stream_url = "http://10.0.0.3:81/stream"

if not video.isOpened():
    print('Not opened')

def video_capture():
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                (flag, encodedImage) = cv2.imencode(".jpg", gray)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

'''
    while (3==3):
        ret, frame = video.read()
        gray = None

        #print(f'Ret = {ret}')

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (flag, encodedImage) = cv2.imencode(".jpg", gray)
            if not flag:
                continue

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
'''

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)

video.release()