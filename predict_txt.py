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
import argparse
import numpy as np
import pandas as pd

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
# NOTE(paddle-dev):所有这些标志都应该是
#设置在“导入桨”之前。否则,它将
#没有任何效果。
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

from reader.task_reader import ClassifyReader
from model.ernie import ErnieConfig
from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("ernie_config_path",            str,  os.path.join(os.getcwd(), 'model', 'ernie_config.json'),  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",              str,  "./checkpoints/step_1033",  "Init checkpoint to resume training from.初始化检查点以恢复训练。")
model_g.add_arg("save_inference_model_path",    str,  "inference_model",  "If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")
model_g.add_arg("num_labels",                   int,  2,     "num labels for classify")
model_g.add_arg("ernie_version",                str,  "1.0", "ernie_version")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("predict_set",         str, os.path.join(os.getcwd(), 'lcqmc', 'test.tsv'),  "Predict set file")
data_g.add_arg("vocab_path",          str, os.path.join(os.getcwd(), 'model', 'vocab.txt'),  "Vocabulary path.")
data_g.add_arg("label_map_config",    str,  None,  "Label_map_config json file.")
data_g.add_arg("max_seq_len",         int,  128,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  50,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   False,  "If set, use GPU for training.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")


# yapf: enable.

class main(object):
    def __init__(self):
        # 载入模型
        args = parser.parse_args()
        model_path = os.path.join(os.getcwd(), 'inference_model', 'step_1_inference_model')
        predict_startup = fluid.Program()
        place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(predict_startup)
        print("load inference model from %s" % model_path)
        self.infer_program, self.feed_target_names, self.probs = fluid.io.load_inference_model(
                model_path, self.exe)

    def predict(self):
        dir_path = os.path.join(os.path.dirname(__file__), "data_http.tsv")
        args = parser.parse_args()
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()

        reader = ClassifyReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            max_seq_len=args.max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=False,
            is_inference=True)
        src_ids = self.feed_target_names[0]
        sent_ids = self.feed_target_names[1]
        pos_ids = self.feed_target_names[2]
        input_mask = self.feed_target_names[3]
        if args.ernie_version == "2.0":
            task_ids = self.feed_target_names[4]

        # 计算相似度
        predict_data_generator = reader.data_generator(
            input_file=dir_path,  #下面的方法写入的路径
            batch_size=args.batch_size,
            epoch=1,
            shuffle=False)


        print("-------------- prediction results --------------")
        np.set_printoptions(precision=4, suppress=True)
        for sample in predict_data_generator():
            src_ids_data = sample[0]
            sent_ids_data = sample[1]
            pos_ids_data = sample[2]
            task_ids_data = sample[3]
            input_mask_data = sample[4]
            if args.ernie_version == "1.0":
                output = self.exe.run(
                    self.infer_program,
                    feed={src_ids: src_ids_data,
                          sent_ids: sent_ids_data,
                          pos_ids: pos_ids_data,
                          input_mask: input_mask_data},
                    fetch_list=self.probs)
            elif args.ernie_version == "2.0":
                output = self.exe.run(
                    self.infer_program,
                    feed={src_ids: src_ids_data,
                          sent_ids: sent_ids_data,
                          pos_ids: pos_ids_data,
                          task_ids: task_ids_data,
                          input_mask: input_mask_data},
                    fetch_list=self.probs)
            else:
                raise ValueError("ernie_version must be 1.0 or 2.0")
            #print(output)
            output_list = []
            for output_temp in output[0]:
                output_list.append(output_temp[1])
            print(output_list)
            return output_list
def txt2tsv(listData):
    '''
    将需要匹配的内容写入文档，后面读取文档进行匹配。
    此处应该直接将list发送给此程序，后续待改写
    '''
    dir_path = os.path.join(os.path.dirname(__file__), "data_http.tsv")  #写入的路径

    #如果文件已存在，则删除原文件
    if os.path.exists(dir_path):
        os.remove(dir_path)

    #加入文件头
    txt_head = [[]]
    txt_head[0].append('text_a')
    txt_head[0].append('text_b')
    #txt_head[0].append('label')
    df_head = pd.DataFrame(txt_head )
    df_head.to_csv(dir_path, sep='\t', mode='a', header=None, index=False )

    #加入内容,删除掉不相关的列
    listData_new = []
    for f in listData:
        i=[]
        i.append(f[1])
        i.append(f[4])
        listData_new.append(i)
    print(listData_new)
    df = pd.DataFrame(listData_new)
    df.to_csv(dir_path, sep='\t', mode='a', header=None, index=False)

def sort(listData,mag_predict):
    for i in range(len(listData)):
        listData[i].append(str(mag_predict[i]))
    def take5(elem):
        return elem[5]
    listData.sort(key=take5, reverse=True)  # 按照第5个值降序排序，返回重新排序的列表
    print(listData)
    return listData[:4]
if __name__ == '__main__':
    #print_arguments(args)
    main()
