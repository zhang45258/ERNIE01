import os
import pandas as pd


def txt2tsv():
    dir_path = os.path.join(os.path.dirname(__file__), "Fine-tuning")
    file_path = os.path.join(os.path.dirname(__file__), "Fine-tuning.tsv")
    if os.path.exists(file_path):
        os.remove(file_path)
    txt_head = [[]]
    txt_head[0].append('text_a')
    txt_head[0].append('text_b')
    txt_head[0].append('label')
    df_head = pd.DataFrame(txt_head )
    df_head.to_csv(file_path, sep='\t', mode='a', header=None, index=False)
    for f in file_name(dir_path):
        txt = pd.read_csv(f, sep='\t')
        txt_new = [[], [], [], []]
        txt_new[0].append(txt.columns[0])
        txt_new[0].append(txt.columns[1])
        txt_new[0].append('1')
        txt_new[1].append(txt.columns[0])
        txt_new[1].append(txt.columns[2])
        txt_new[1].append('0')
        txt_new[2].append(txt.columns[0])
        txt_new[2].append(txt.columns[3])
        txt_new[2].append('0')
        txt_new[3].append(txt.columns[0])
        txt_new[3].append(txt.columns[4])
        txt_new[3].append('0')
        df = pd.DataFrame(txt_new)
        print(df)
        df.to_csv(file_path, sep='\t', mode='a', header=None, index=False)


#获取文件夹下所有txt格式 的文件名
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L


if __name__ == '__main__':
    txt2tsv()