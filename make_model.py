#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Load classifier's checkpoint to do prediction or save inference model.
加载分类器的检查点做预测或保存推理模型。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
# NOTE(paddle-dev):所有这些标志都应该是
#设置在“导入桨”之前。否则,它将
#没有任何效果。
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid
from model.ernie import ErnieConfig
from finetune.classifier import create_model
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params



# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("ernie_config_path",            str,  os.path.join(os.getcwd(), 'model', 'ernie_config.json'),  "Path to the json file for bert model config.")
#model_g.add_arg("init_checkpoint",              str,  "./checkpoints/step_1033",  "Init checkpoint to resume training from.初始化检查点以恢复训练。")
model_g.add_arg("save_inference_model_path",    str,  "inference_model",  "If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")
model_g.add_arg("num_labels",                   int,  2,     "num labels for classify")
model_g.add_arg("ernie_version",                str,  "1.0", "ernie_version")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("predict_set",         str, os.path.join(os.getcwd(), 'lcqmc', 'test.tsv'),  "Predict set file")
data_g.add_arg("vocab_path",          str, os.path.join(os.getcwd(), 'model', 'vocab.txt'),  "Vocabulary path.")
data_g.add_arg("label_map_config",    str,  None,  "Label_map_config json file.")
data_g.add_arg("max_seq_len",         int,  128,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   False,  "If set, use GPU for training.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")

args = parser.parse_args()
# yapf: enable.

def main(args,init_checkpoint):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            predict_pyreader, probs, feed_target_names = create_model(
                args,
                pyreader_name='predict_reader',
                ernie_config=ernie_config,
                is_classify=True,
                is_prediction=True,
                ernie_version=args.ernie_version)

    predict_prog = predict_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(predict_startup)

    if init_checkpoint:
        init_pretraining_params(exe, init_checkpoint, predict_prog)
    else:
        raise ValueError("args 'init_checkpoint' should be set for prediction!")

    #保存模型
    assert args.save_inference_model_path, "args save_inference_model_path should be set for prediction"
    _, ckpt_dir = os.path.split(init_checkpoint.rstrip('/'))
    dir_name = ckpt_dir + '_inference_model'
    model_path = os.path.join(args.save_inference_model_path, dir_name)
    print("save inference model to %s" % model_path)
    fluid.io.save_inference_model(
        model_path,
        feed_target_names, [probs],
        exe,
        main_program=predict_prog)


if __name__ == '__main__':
    from lcqmc.txt2tsv import txt2tsv
    txt2tsv()
    print("文件转换完成")

    from run_classifier_lcqmc import main as mn
    path = mn()
    print("Fine-tuning任务完成")

    print_arguments(args)
    main(args=args, init_checkpoint=path)
    print("模型生成完成")

