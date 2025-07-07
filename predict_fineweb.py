import fasttext
import datasets
import itertools
import re
import regex

FINE_WEB_NUM = 5000
# 这里是我们训练时下载过的fineweb条目数
NON_MATH_CLASS_NUM = 60000
MODEL_PATH = "math_classifier.bin"
RESULT_PATH = "fineweb_5000_predictions.txt"

def main():
    """
    此脚本会从Hugging Face流式下载5000条数据，并使用训练好的模型math_classifier.bin
    进行分类预测，结果保存在fineweb_5000_predictions.txt中。
    """
    fine_web_stream = datasets.load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2014-10",streaming=True,split="train")
    fine_web_iter = itertools.islice(fine_web_stream,FINE_WEB_NUM + NON_MATH_CLASS_NUM)
    model = fasttext.load_model(MODEL_PATH)
    count = 0
    file_out = open(RESULT_PATH,'w',encoding='utf-8')
    for fine_web_dict in fine_web_iter:
        count+=1
        if count < NON_MATH_CLASS_NUM:
            continue # 跳过训练用的fineweb条目
        if not "text" in fine_web_dict:
            continue
        text = fine_web_dict.get("text")
        text = re.sub(r'\s+',' ',text).strip()
        text = regex.sub(r'\p{C}',' ',text)
        result = model.predict(text)[0][0] # 提取预测结果中的标签
        file_out.write(result + ' ' + text + '\n') # 整理格式后写入输出文件
    file_out.close()

if __name__ == "__main__":
    main()