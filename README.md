本代码是论文开源代码的一部分。本代码用于执行ERNIE 2.0的Fine-tuning任务后，生成 model。
操作步骤：
1.将所有“正反例”txt文本放在".lcqml\Fine-tuning"位置（清空原有位置的文件）。
2.将finetune_args.py第29行：model_g.add_arg("init_checkpoint", str,  "./checkpoints/step_****", ......）地址修改为"./checkpoints/step_****"(上次任务生成文件夹名称)。要使用上次训练的数据继续训练。如果不改，上一次的训练数据就没有存储。
3.执行make_model.py
  (1)将正反例文本以规定格式保存在".lcqml\Fine-tuning.tsv"中。
  (2)执行Fine-tuning任务。
  (3)生成model并保存。
