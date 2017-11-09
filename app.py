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

import os, uuid, json

from flask import Flask, request, url_for, render_template, send_file
from auth import requires_auth
from im2txt_inference import CaptionResult, im2txt_inference

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pwd = os.path.dirname(os.path.realpath(__file__))

im2txt_inst = im2txt_inference(os.path.join(pwd, 'chkpt', 'model.ckpt-1000000'),
        os.path.join(pwd, 'chkpt','word_counts.txt'))

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.before_request
@requires_auth
def before_request():
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image():
    imagefile = request.args['name']
    return send_file('/tmp/' + imagefile, mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = '%s.%s' % (str(uuid.uuid4()), file.filename.rsplit('.', 1)[1])
            fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fullpath)
            results = im2txt_inst.inference(fullpath)
            return """
                <script language="javascript" type="text/javascript">window.top.window.inferenceDone('%s', '%s');</script>
                """ % ((url_for('image') + '?name=' + filename), json.dumps(results))
    return ''

port = os.getenv('PORT', '5000')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port))
