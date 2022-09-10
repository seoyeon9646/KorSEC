# SEC : Space Error Correction
from collections import defaultdict
from KorSEC.sec import SEC
from KorSEC import util

import plyvel
import random
import pickle
import tqdm

class Trainer:
    def __init__(self, tuning_data_ratio=0.1, min_count=5):
        """
        KorSEC : Korean Space Error Correction Trainer

        args:
            tuning_data_ratio[float, default=0.1]
                - 가중치 튜닝에 사용할 데이터 양
                - 전체 데이터 * tuning_data_ratio만큼 튜닝에 사용합니다.
            min_count[int, default=5]
                - 임의의 N-gram 음절에 최소 등장 횟수
                - 학습 데이터에서 min_count보다 적은 횟수 등장하면 버려집니다.
        """
        unigram = defaultdict(list)
        bigram = defaultdict(list)
        trigram = defaultdict(list)

        self.ngram_cnt_dict = {
            1 : unigram,
            2 : bigram,
            3 : trigram
        }
        self.SEC = SEC(mode="train")
        self.learning_rate = 0.0001
        self.min_count = min_count
        self.tuning_data_ratio = tuning_data_ratio
        self.start, self.end = "S", "E"

    def _ngram_slicing(self, sent_wo_space:str, label:str, n:int) -> None:
        """
        N-gram 단위로 문장을 나누어 각 음절에 대한 띄어쓰기 패턴을 수집합니다.

        args:
            sent_wo_space[str]
                - 사용자가 입력한 문장의 띄어쓰기 제거 버전
            label[str]
                - 사용자가 입력한 띄어쓰기의 위치 정보을 표현한 레이블
                - 1은 띄어쓰기 있음, 0은 띄어쓰기 없음
            n[int]
                - N-gram 단위
        """
        for i in range(len(sent_wo_space)-n+1):
            key = sent_wo_space[i:i+n]
            try:
                space_info = int("0b" + label[i:i+n+1], 2)
            except:
                continue
            if key not in self.ngram_cnt_dict[n]:
                initial_freq = [0]*(2**(n+1))
                self.ngram_cnt_dict[n][key] = initial_freq
            self.ngram_cnt_dict[n][key][space_info]+=1

    def train(self, train_file_path:str, model_path:str) -> None:
        """
        한국어 문장으로 이루어진 텍스트 파일을 이용하여 학습
        
        args:
            train_file_path[str]
                - 학습에 사용할 텍스트 파일
            model_path[str]
                - 학습한 정보를 저장할 경로
        """
        corpus = []
        with open(train_file_path, "r", encoding="utf-8") as f:
            print(f"TRAINING START WITH '{train_file_path}'")
            for sentence in tqdm.tqdm(f.readlines()):
                sentence = sentence.replace("\n", "").replace(" ", " ").strip()
                _, prep_sen = util.prep(sentence)
                corpus.append(prep_sen)

                label = "1" + self.SEC.labeling(prep_sen) + "11"
                prep_sen = self.start + prep_sen + self.end
                sent_wo_space = prep_sen.replace(" ", "")
                if len(label)<10:
                    continue
                elif len(label)!=(len(sent_wo_space)+1):
                    continue

                self._ngram_slicing(sent_wo_space, label, 1)
                self._ngram_slicing(sent_wo_space, label, 2)
                self._ngram_slicing(sent_wo_space, label, 3)

        print("TOTAL VOCABS")
        print("UNIGRAM = ", len(self.ngram_cnt_dict[1]))
        print("BIGRAM = ", len(self.ngram_cnt_dict[2]))
        print("TRIGRAM = ", len(self.ngram_cnt_dict[3]))
        print("-"*30)

        self.SEC.set_freq_dict_from_trainer(self.ngram_cnt_dict)

        print(f"WEIGHT TUNING...")
        random.shuffle(corpus)
        splited = corpus[:int(len(corpus)*self.tuning_data_ratio)]
        tuned_weight = self.weight_tuning(splited)
        print("WEIGHT TUNING DONE")
        print(tuned_weight)
        self._save_all(model_path, tuned_weight)
    
    def _save_all(self, file_path:str, tuned_weight:dict) -> None:
        # 학습 파일을 저장
        self._save_db(file_path)
        self._save_to_weight(file_path, tuned_weight)
    
    def _save_db(self, file_path:str) -> None:
        # N-gram count dictionry를 DB로 저장(levelDB)
        marker_dict = {1:"u", 2:"b", 3:"t"}
        db = plyvel.DB(file_path+".dict", create_if_missing=True)
        wb = db.write_batch()

        for n, ndict in self.ngram_cnt_dict.items():
            for k, v in ndict.items():
                if sum(v)<self.min_count:
                    continue
                key = (marker_dict[n] + k).encode()
                val = str(v).encode()
                wb.put(key, val)
        wb.write()

    def _save_to_weight(self, file_path:str, tuned_weight:dict) -> None:
        # 튜닝한 파라미터 정보를 저장
        with open(file_path+".param", "wb") as f:
            pickle.dump(tuned_weight, f)

    def weight_update(self, correct:str, predicts:list, weights:list):
        """
        띄어쓰기 확률 기어도에 따라 가중치를 업데이트

        args:
            correct[str]
                - 실제 띄어쓰기 유무
                - 1은 띄어쓰기 있음, 0은 띄어쓰기 없음
            predicts[list]
            weights[list]
        """
        new_weights = []
        for predict, w in zip(predicts, weights):
            error = abs(int(correct) - predict)
            w_new = w - self.learning_rate * (error - 0.5)
            new_weights.append(w_new)

        return new_weights

    def weight_adjust(self, weight:list) -> list:
        """
        가중치 범위 재조정. 가중치의 합이 1이 되도록 조정합니다.
        """
        weight_sum = sum(weight)
        adjusted = [w/weight_sum for w in weight]

        return adjusted

    def sum_of_product(self, weight:list, prob:list) -> float:
        """
        확률의 가중합 계산
        """
        total = 0
        for w, p in zip(weight, prob):
            total += (w*p)
        return total

    def weight_tuning(self, corpus:list) -> dict:
        """
        띄어쓰기 확률 계산에 사용하는 파라미터 튜닝(KorSEC.SEC에서 사용)
        """
        space_prob_dict = defaultdict(list)
        bigram_weights = self.SEC.get_bigram_weights()
        trigram_weights = self.SEC.get_trigram_weights()
        prob_weights = self.SEC.get_prob_weight()

        for sentence in tqdm.tqdm(corpus):
            sentence = sentence.replace("\n", "").replace(" ", "").strip()
            _, prep_sen = util.prep(sentence)
            if not sentence:
                continue

            label = "1" + self.SEC.labeling(prep_sen) + "11"
            prep_sen = self.start + prep_sen + self.end
            sent_wo_space = prep_sen.replace(" ", "")

            bigram_p = self.SEC.bigram_prob_for_tune(sent_wo_space)
            trigram_p = self.SEC.trigram_prob_for_tune(sent_wo_space)
            
            if len(label)<=len(bigram_p) or len(label)!=(len(sent_wo_space)+1):
                continue

            for i in range(len(bigram_p)):
                correct = label[i+1]
                bigram_weights = self.weight_update(correct, bigram_p[i], bigram_weights)
                trigram_weights = self.weight_update(correct, trigram_p[i], trigram_weights)
                
                b_prob = self.sum_of_product(self.weight_adjust(bigram_weights), bigram_p[i])
                t_prob = self.sum_of_product(self.weight_adjust(trigram_weights), trigram_p[i])

                prob_weights = self.weight_update(correct, [t_prob, b_prob], prob_weights)
                space_prob = self.sum_of_product(self.weight_adjust(prob_weights), [t_prob, b_prob])
                space_prob_dict[correct].append(space_prob)

        bigram_weights = self.weight_adjust(bigram_weights)
        trigram_weights = self.weight_adjust(trigram_weights)
        prob_weights = self.weight_adjust(prob_weights)

        avg_space_prob = sum(space_prob_dict["1"])/len(space_prob_dict["1"])
        avg_paste_prob = sum(space_prob_dict["0"])/len(space_prob_dict["0"])

        spacing_threshold = round((avg_space_prob+avg_paste_prob)/2, 4)
        pasting_threshold = round(2*avg_paste_prob, 4)

        trained_weights = {
            "bigram_weight" : bigram_weights,
            "trigram_weight" : trigram_weights,
            "prob_weight" : prob_weights,
            "spacing_threshold" : spacing_threshold,
            "pasting_threshold" : pasting_threshold
        }
        return trained_weights


if __name__ == "__main__":
    sec = Trainer()
    sec.train("./data_in/modu_dump.txt", "./sec_modu_params")