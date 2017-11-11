# Copyright 2017 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from flask import Flask, request, render_template, send_file
from auth import requires_auth
from im2txt_inference import ShowAndTellInference

APP = Flask(__name__)
PWD = os.path.dirname(os.path.realpath(__file__))

MODEL_INST = ShowAndTellInference(
    os.path.join(PWD, 'chkpt', 'model.ckpt-1000000'),
    os.path.join(PWD, 'chkpt', 'word_counts.txt'))

@APP.before_request
@requires_auth
def before_request():
    pass

@APP.route('/')
def index():
    """
    Serve the index page
    """
    return render_template('index.html')

@APP.route('/image/<image_file>')
def image(image_file):
    """
    Serve the image request

    Retrieve the image from tmp and send back to client
    """
    if not image_file:
        return

    path = os.path.join('/tmp', image_file)
    val = send_file(path, mimetype='image/jpeg')
    print 'remote file:%s' % path 
    os.remove(path)
    return val

@APP.route('/inference', methods=['POST'])
def inference():
    """
    Serve the inference request

    Store the image into tmp and run inference against this image
    """
    if request.method == 'POST':
        return MODEL_INST.inference(request)
    return ''

PORT = os.getenv('PORT', '5000')
if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=int(PORT))
