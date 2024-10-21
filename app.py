
from flask import Flask, render_template, Response, request, jsonify

import cv2
import json
import uuid
import asyncio
import logging
import time
app = Flask(__name__)


@app.route('/')
def index():
    return 'Hola Mundo!!!'


if __name__=="__main__":
    app.run()


