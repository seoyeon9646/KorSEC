# KorSEC : Korean Space Error Correction
N-gram 확률 기반의 한국어 띄어쓰기 및 붙여쓰기 오류 교정 모델
(아직 성능 개선 단계라서 교정이 제대로 되지 않을 수 있어요😂)


## Introduction
 기존에는 띄어쓰기 교정 방식은 입력된 문장의 띄어쓰기 정보를 모두 삭제한 후에 다시 공백을 추가하는 방식이 일반적인 방식이었습니다. 하지만 이 경우에는 기존의 문장의 의도를 반영하지 못 한다는 문제점이 존재합니다. KorSEC는 유저가 입력한 띄어쓰기 정보를 반영하여 교정에 활용함으로써 딥러닝, 머신러닝 기반의 모델에 준하는 성능을 보이는 확률 기반의 모델입니다. (띄어쓰기 정보를 하나도 추가하지 않을 경우에는 비교적 낮은 성능을 보일 수도 있습니다.)
 - 아빠가 방에서 서류봉투를 꺼냈다.
 - 아빠 가방에서 서류봉투를 꺼냈다.
 
 현재 python package에 포함되어 있는 모델 파일은 용량 문제로 인해 기존 데이터를 모두 사용하지 못 한 경량 버전의 모델이기 때문에 성능이 낮을 수 있습니다. 이 부분에 대해서는 추후에 보완하여 모델 파일을 업데이트 할 예정입니다.

## Install
KorSEC는 아래 두 가지 방법으로 시작하실 수 있습니다.

1. pip 명령어로 시작하기(경량모델 포함)
```sh
pip install KorSEC
```

2. Github에서 clone하여 시작하기(모델 미포함)
```sh
git clone https://github.com/seoyeon9646/KorSEC.git
```

## Train
현재 repo에 포함되어있는 `main.py`와 `tutorial.ipynb`(업로드 예정)를 참고해주세요🥳
```python
from KorSEC import Trainer
sec_trainer = Trainer()
sec_trainer.train("학습에 사용할 데이터파일.txt", "모델 파일의 이름")
```
- 학습이 정상적으로 끝나면 직접 지정해주신 "모델 파일의 이름".dict 폴더와 "모델 파일의 이름".param 파일이 생성됩니다.

## Correction
현재 repo에 포함되어있는 `main.py`와 `tutorial.ipynb`(업로드 예정)를 참고해주세요🥳
```python
from KorSEC import SEC
sec = SEC("모델 파일 이름")
sec.correction("아빠가 방에서서류봉 투를꺼냈 다.")
# > 아빠가 방에서 서류봉투를 꺼냈다.
```
- PyPI로 패키지를 설치하고, 모델 파일 이름을 따로 설정하지 않을 경우에는 패키지에 포함되어 있는 경량 모델로 교정을 진행합니다.


## 📞
seoyeon9695@gmail.com
