import fasttext
import os

def main():
    dir_name = "data"
    train_text_name = "train.txt"
    valid_text_name = "valid.txt"
    model = fasttext.train_supervised(
        os.path.join(dir_name,train_text_name),
        wordNgrams = 2,
        autotuneValidationFile = os.path.join(dir_name,valid_text_name),
        autotuneDuration=600
    )

    model.save_model("math_classifier.bin")

if __name__ == "__main__":
    main()