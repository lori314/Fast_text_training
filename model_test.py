import fasttext
import os

MODEL_PATH = "math_classifier.bin"
TEST_FILE_NAME = "test.txt"
DATA_DIR = "data"

def main():
    model = fasttext.load_model(MODEL_PATH)
    result = model.test(os.path.join(DATA_DIR,TEST_FILE_NAME))
    print(f"测试集样本数量为{result[0]}")
    print(f"模型的精确率为：{result[1]:.4f}")
    print(f"模型的召回率为:{result[2]:.4f}")
       

if __name__ == "__main__":
    main()
