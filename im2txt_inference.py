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

""" Wrap Show and Tell inference"""

import math
import os
import json
import uuid

from PIL import Image
from flask import url_for

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = '/tmp'

class ShowAndTellInference(object):
    """
    image 2 text inference class
    """

    def __init__(self, checkpoint, vocab_file):
        tf.logging.set_verbosity(tf.logging.INFO)

        if not os.path.isfile(vocab_file):
            raise Exception('can not find vocabulary file')
        self.checkpoint = checkpoint
        self.vocab_file = vocab_file

        # Build the inference graph.
        self.g = tf.Graph()
        with self.g.as_default():
            self.model = inference_wrapper.InferenceWrapper()
            restore_fn = self.model.build_graph_from_config(configuration.ModelConfig(),
                                                self.checkpoint)
        self.g.finalize()
        # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary(self.vocab_file)

        self.session = tf.Session(graph=self.g)
        restore_fn(self.session)
        self.generator = caption_generator.CaptionGenerator(self.model, self.vocab)

    def inference(self, request):
        """
        run inference against the img_file
        """
        img_file = request.files['image']
        if img_file and allowed_file(img_file.filename):
            image_path, filename = save_image_as_jpeg(img_file)
            results = []
            with tf.gfile.GFile(image_path, 'r') as f:
                image = f.read()

            captions = self.generator.beam_search(self.session, image)
            for __, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = ' '.join(sentence)
                results.append({'caption': sentence, 'p': math.exp(caption.logprob)})

            return """
                <script language="javascript" type="text/javascript">window.top.window.inferenceDone('%s', '%s');</script>
                """ % (url_for('image', image_file=filename), json.dumps(results))
        else:
            return """
                <script language="javascript" type="text/javascript">window.top.window.inferenceFailed('%s');</script>
                """ % 'Unsupported file type'


    def close(self):
        """
        Close and release the resources
        """
        if (self.session):
            self.session.close()

def allowed_file(filename):
    """
    Check the image extension

    Currently, only support jpg, jpeg and png
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def save_image_as_jpeg(img_file):
    """
    Save image file (FileStore) from request and save it to
    upload folder
    If it's a png file, convert it to jpeg

    TODO: maybe do convertion in memeory
    """
    extension = img_file.filename.rsplit('.', 1)[1]
    name = str(uuid.uuid4())
    filename = '%s.%s' % (name, extension)
    fullpath = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(fullpath)
    if extension == 'png':
        png = Image.open(fullpath)
        rgb_im = png.convert('RGB')
        os.remove(fullpath) # remove png file
        filename = '%s.%s' % (name, 'jpg')
        fullpath = os.path.join(UPLOAD_FOLDER, filename)
        rgb_im.save(fullpath)

    return fullpath, filename

