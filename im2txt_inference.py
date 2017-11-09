# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from collections import namedtuple

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

CaptionResult = namedtuple('CaptionResult', 'caption p')

class im2txt_inference(object):
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

  def inference(self, img_file):
    with tf.gfile.GFile(img_file, "r") as f:
      image = f.read()
    
    results = []
    captions = self.generator.beam_search(self.session, image)
    print("Captions for image %s:" % os.path.basename(img_file))
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      results.append({'caption': sentence, 'p': math.exp(caption.logprob)})

    return results

  def closeSession(self):
    if (self.session):
      self.session.close()
