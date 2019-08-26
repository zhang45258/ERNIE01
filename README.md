## 客服中心智能问答系统二次匹配服务组件（包括模型生成）  
    本代码是论文开源代码的一部分。本代码的执行环境：win64， python3.5+， PaddlePaddle框架（百度飞桨）    
   ### 本代码用于：  
 ### 一、执行Fine-tuning任务，生成 model。  
    操作步骤：  
      1.将所有“正反例”txt文本放在".lcqml\Fine-tuning"位置（清空原有位置的文件）。  
      2.将finetune_args.py第29行：model_g.add_arg("init_checkpoint", str,  "./checkpoints/step_****", ......）  
      地址修改为"./checkpoints/step_****"(上次任务生成文件夹名称)。要使用上次训练的检查点继续训练。如果不改，上一次的训练  
      数据就没有存储。这里提供一个检查点的下载地址：  
      3.执行make_model.py：（该程序执行以下操作）  
        (1)将正反例文本以规定格式保存在"./lcqml/Fine-tuning.tsv"中。  
        (2)执行Fine-tuning任务。  
        (3)生成model并保存。（保存地址：./inference_model/）  
### 二、生成http接口，接收首次匹配的数据，返回二次匹配的数据  
    操作步骤：  
      1.执行Api_http.py（该程序执行以下操作）  
        (1)生成http接口。接收内容：post方式，json数据，{"txt":"客户问话原文"，"listData":"首次匹配结果list"}，返回内容：  
        json数据，{"listData":"二次匹配结果list"}。  
        （2）接收首次匹配服务组件发送json数据，解析json，将内容按规定格式写入到"./data_http.tsv"中。  
        （3）读取"./data_http.tsv"文档，使用model进行语义相似度计算。  
        （4）计算结果进行排序，截取前4名  
        （5）序列化成json数据并返回。
