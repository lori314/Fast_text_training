from datasets import load_dataset
import itertools
import re
import os
import regex

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
    if not os.path.exists(DATA_DIR_NAME):
        os.mkdir(DATA_DIR_NAME)

    math_stream = load_dataset("open-web-math/open-web-math",streaming=True,split="train")
    limit_math_stream = itertools.islice(math_stream,MATH_CLASS_NUM)
    non_math_stream = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2014-10",streaming=True,split="train")
    limit_non_math_stream = itertools.islice(non_math_stream,NON_MATH_CLASS_NUM)

    f_train = open(os.path.join(DATA_DIR_NAME,TRAIN_FILE_NAME),"w",encoding='utf-8')
    f_valid = open(os.path.join(DATA_DIR_NAME,VALID_FILE_NAME),"w",encoding='utf-8')
    f_test = open(os.path.join(DATA_DIR_NAME,TEST_FILE_NAME),"w",encoding='utf-8')
    print("开始下载正样例")
    count = 0
    for sub_math_dict in limit_math_stream:
        count+=1
        if not "text" in sub_math_dict:
            continue
        text = sub_math_dict["text"]
        text = re.sub(r'\s+',' ',text).strip()
        text = regex.sub(r'\p{C}',' ',text)
        data = "__label__math " + text + '\n'
        if count <= MATH_CLASS_NUM * TRAIN_SPLIT:
            f_train.write(data)
        elif count > MATH_CLASS_NUM * TRAIN_SPLIT and count <= MATH_CLASS_NUM * (TRAIN_SPLIT + VALID_SPLIT):
            f_valid.write(data)
        else:
            f_test.write(data)
    print("正样例下载完毕，开始下载负样例")
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

if __name__ == "__main__":
    main()