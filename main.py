import os

from KorSEC import Trainer, SEC


def train(args):
    if (args.train_file is None) or (not os.path.exists(args.train_file)):
        raise Exception("Train file does not exits!")
    
    dict_name = args.dict_path
    if dict_name is None:
        dict_name = "./data/sec"
    trainer = Trainer(args.tuning_data_ratio)
    trainer.train(args.train_file, dict_name)

def pred(args):
    sec = SEC(args.dict_path)
    while True:
        sent = input("input sentence = ").strip()
        print(sec.correction(sent))

def main(args):
    if args.train:
        train(args)
    
    elif args.infer:
        pred(args)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Korean Space Error Correction")
    # trainer에서 사용할 학습 파일
    parser.add_argument("--train_file", type=str, default=None, help="Training file name for Kospacing")
    # trainer가 학습한 후에 저장할 dict 파일의 경로
    parser.add_argument("--dict_path", type=str, default=None, help="Kospacing dict path")
    # 튜닝할 때 사용할 데이터의 비율(Train file * ratio)
    parser.add_argument("--tuning_data_ratio", type=float, default=0.1, help="Tuning data ratio for weight tuning")

    parser.add_argument("--train", action="store_true", help="Whether to run training.")
    parser.add_argument("--infer", action="store_true", help="Whether to run correction.")

    args = parser.parse_args()
    main(args)
