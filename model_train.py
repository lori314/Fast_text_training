import fasttext
import os

def main():
    """
    此脚本会读取data/train.txt和data/valid.txt进行模型训练和自动调参，并将最优模型保存为math_classifier.bin。
    """
    # 设置训练数据
    dir_name = "data"
    train_text_name = "train.txt"
    valid_text_name = "valid.txt"

    # 训练模型
    model = fasttext.train_supervised(
        os.path.join(dir_name,train_text_name),
        wordNgrams = 2,
        autotuneValidationFile = os.path.join(dir_name,valid_text_name),
        autotuneDuration=600
    )

    # 保存模型
    model.save_model("math_classifier.bin")

if __name__ == "__main__":
    main()