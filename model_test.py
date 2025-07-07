import fasttext
import os

MODEL_PATH = "math_classifier.bin"
TEST_FILE_NAME = "test.txt"
DATA_DIR = "data"

def main():
    """
    此脚本会加载data/test.txt和训练好的模型math_classifier.bin，并返回测试结果，展示为精确率和召回率
    """
    # 使用fasttext自带的model.test来获取预测结果
    model = fasttext.load_model(MODEL_PATH)
    result = model.test(os.path.join(DATA_DIR,TEST_FILE_NAME))
    print(f"测试集样本数量为{result[0]}")
    print(f"模型的精确率为：{result[1]:.4f}")
    print(f"模型的召回率为:{result[2]:.4f}")
       

if __name__ == "__main__":
    main()
