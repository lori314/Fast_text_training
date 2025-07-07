from datasets import load_dataset
import itertools
import re
import os
import regex

# 定义下载路径和条目数,以及划分比例
DATA_DIR_NAME = "data"
MATH_CLASS_NUM = 40000
NON_MATH_CLASS_NUM = 60000
TRAIN_FILE_NAME = "train.txt"
TRAIN_SPLIT = 0.8
VALID_FILE_NAME = "valid.txt"
VALID_SPLIT = 0.15
TEST_FILE_NAME = "test.txt"
TEST_SPLIT = 0.05

def main():
    """
    此脚本会从Hugging Face流式下载正负样本，进行数据清洗，并按照设定好的的比例切分，生成data/目录下的三个.txt文件。
    """
    if not os.path.exists(DATA_DIR_NAME):
        os.mkdir(DATA_DIR_NAME)

    # 流式加载数据集
    math_stream = load_dataset("open-web-math/open-web-math",streaming=True,split="train")
    # 使用迭代器工具进行切片
    limit_math_stream = itertools.islice(math_stream,MATH_CLASS_NUM)
    non_math_stream = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2014-10",streaming=True,split="train")
    limit_non_math_stream = itertools.islice(non_math_stream,NON_MATH_CLASS_NUM)

    f_train = open(os.path.join(DATA_DIR_NAME,TRAIN_FILE_NAME),"w",encoding='utf-8')
    f_valid = open(os.path.join(DATA_DIR_NAME,VALID_FILE_NAME),"w",encoding='utf-8')
    f_test = open(os.path.join(DATA_DIR_NAME,TEST_FILE_NAME),"w",encoding='utf-8')
    print("开始下载正样例")

    # 处理正样例
    count = 0
    for sub_math_dict in limit_math_stream:
        count+=1
        if not "text" in sub_math_dict:
            continue
        # 数据清洗，提取text字段，删除其中的换行符使其为一行以符合训练数据格式，并删除其中的控制字符
        text = sub_math_dict["text"]
        text = re.sub(r'\s+',' ',text).strip()
        text = regex.sub(r'\p{C}',' ',text)
        # 按格式拼接
        data = "__label__math " + text + '\n'
        if count <= MATH_CLASS_NUM * TRAIN_SPLIT:
            f_train.write(data)
        elif count > MATH_CLASS_NUM * TRAIN_SPLIT and count <= MATH_CLASS_NUM * (TRAIN_SPLIT + VALID_SPLIT):
            f_valid.write(data)
        else:
            f_test.write(data)
    print("正样例下载完毕，开始下载负样例")

    # 处理负样例
    count = 0
    for sub_non_math_dict in limit_non_math_stream:
        count+=1
        if not "text" in sub_non_math_dict:
            continue
        text = sub_non_math_dict["text"]
        text = re.sub(r'\s+',' ',text).strip()
        data = "__label__non_math " + text + '\n'
        if count <= NON_MATH_CLASS_NUM * TRAIN_SPLIT:
            f_train.write(data)
        elif count > NON_MATH_CLASS_NUM * TRAIN_SPLIT and count <= NON_MATH_CLASS_NUM * (TRAIN_SPLIT + VALID_SPLIT):
            f_valid.write(data)
        else:
            f_test.write(data)
    print("负样例下载完毕，全部下载完成")

    # 关闭文件
    f_test.close()
    f_train.close()
    f_valid.close()

if __name__ == "__main__":
    main()