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

def concat_chkp(path, checkpoint_prefix):
    "concat check point files"
    pwd = os.path.dirname(os.path.realpath(__file__))
    files = []

    index = 1
    target = os.path.join(pwd, path, checkpoint_prefix)
    while True:
        tmp = '%s_%04d' % (target, index)

        if os.path.isfile(tmp):
            files.append(tmp)
            index += 1
        else:
            break
                
    if not files:
        print('no file to concat')
        return

    with open(target, 'wb') as target_f:
        for a_file in files:
            with open(a_file, 'rb') as one_f:
                while True:
                    data = one_f.read(65536)
                    if not data:
                        break
                    target_f.write(data)

if __name__ == "__main__":
    concat_chkp('chkpt', 'model.ckpt-1000000.data-00000-of-00001')