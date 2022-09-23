import os
import ast
import pickle
import plyvel

from KorSEC import util
from collections import defaultdict

class SEC:
    def __init__(self, dict_name = None, mode="fast"):
        self.alpha = 0.001
        self.min_prob, self.max_prob = 1e-5, 1-(1e-05)
        self.bi_target_idx_dict ={
            0 : [4, 5, 6, 7],   # left
            1 : [2, 3, 6, 7],   # middle
            2  : [1, 3, 5, 7]   # right
        }
        self.tri_target_idx_dict =  {
            0 : list(range(8, 16)), #left
            1 : list(range(4, 8)) + list(range(12, 16)), #mid left
            2 : [2, 3, 6, 7, 10, 11, 14, 15], #mid right
            3 : list(range(1, 16, 2)) # right
        }
        self.marker_dict = {1:"u", 2:"b", 3:"t"}
        
        self.cache_dict = defaultdict(list)
        self.start, self.end = "S", "E"
        self.w1, self.w2 = 0.537, 0.463
        self.b1, self.b2, self.b3 = 0.302, 0.376, 0.322
        self.t1, self.t2, self.t3, self.t4 = 0.216, 0.272, 0.274, 0.238
        self.pasting_threshold, self.weak_pasting_threshold = 0.1, 0.1376
        self.spacing_threshold, self.weak_spacing_threshold = 0.5, 0.4485
        self.get_freq_method = self.get_freq_from_dict if mode in ["fast", "train"] else self.get_freq_from_db
        
        if dict_name == None:
            dict_name = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data/sec"
            )

        if mode =="fast":
            self.load_freq_dict(dict_name)
        elif mode !="train":
            self.load_db(dict_name)

    def load_freq_dict(self, dict_name):
        if not os.path.exists(dict_name+".freq"):
            raise Exception("Model doesn't exists! Train first!")
        else:
            with open(dict_name + ".freq", "rb") as f:
                self.ngram_cnt_dict = pickle.load(f)
            #self.load_parameters(dict_name)

    def load_db(self, dict_name):
        if not os.path.exists(dict_name+".dict"):
            raise Exception("Model doesn't exists! Train first!")

        self.db = plyvel.DB(dict_name+".dict")  
        self.load_parameters(dict_name)  

    def load_parameters(self, dict_name):
        if not os.path.exists(dict_name+".param"):
            print("Param file doesn't exists! We'll use default parameters")
        else:
            with open(dict_name + ".param", "rb") as f:
                weight_dict = pickle.load(f)
            self.set_weights(weight_dict)

    def set_weights(self, weight_dict):
        '''
        trained_weights = {
            "bigram_weight" : bigram_weights,
            "trigram_weight" : trigram_weights,
            "prob_weight" : prob_weights,
            "spacing_threshold" : spacing_threshold,
            "pasting_threshold" : pasting_threshold
        }
        '''
        self.update_bigram_weight(weight_dict["bigram_weight"])
        self.update_trigram_weight(weight_dict["trigram_weight"])
        self.update_prob_weight(weight_dict["prob_weight"])

    def set_freq_dict_from_trainer(self, ngram_cnt_dict):
        self.ngram_cnt_dict = ngram_cnt_dict

    def update_bigram_weight(self, new_weights):
        self.b1, self.b2, self.b3 = new_weights

    def update_trigram_weight(self, new_weights):
        self.t1, self.t2, self.t3, self.t4 = new_weights

    def update_prob_weight(self, new_weights):
        self.w1, self.w2 = new_weights

    def get_bigram_weights(self):
        return self.b1, self.b2, self.b3

    def get_trigram_weights(self):
        return self.t1, self.t2, self.t3, self.t4

    def get_prob_weight(self):
        return self.w1, self.w2

    def labeling(self, sentence:str) -> str:
        label = ""
        for word in sentence.split():
            label = label + "1" + "0"*(len(word)-1)
        return label

    def _sum_all(self, values:list):
        result = sum(values)
        if result !=0:
            return result
        return (result + self.alpha)

    def _sum_elements(self, values:list, indexs:list):
        result = 0
        for i in indexs:
            result += values[i]
        if result != 0 :
            return result
        return (result + self.alpha)

    def get_freq_from_db(self, key, depth=0):
        if key in self.cache_dict:
            return self.cache_dict[key]

        length = len(key)
        marker = self.marker_dict[length]
        encoded_key = (marker + key).encode()

        result = self.db.get(encoded_key)

        if result is None:
            if (length == 1) or (depth>2):
                self.cache_dict[key] = [0]*(2**(len(key)+1))
                return [0]*(2**(len(key)+1))
            return self.get_indirect_prob(key, depth)
        
        val = ast.literal_eval(result.decode())
        self.cache_dict[key] = val
        return val
    
    def get_freq_from_dict(self, key, depth=0):
        length = len(key)
        if key not in self.ngram_cnt_dict[length]:
            if (length == 1) or (depth>2):
                return [0]*(2**(len(key)+1))
            return self.get_indirect_prob(key, depth)
        return self.ngram_cnt_dict[length][key]

    def get_indirect_prob(self, key:str, depth:int) -> list:
        result = []
        src_n, tar_n = len(key)-1, len(key)
        k1, k2 = key[:src_n], key[src_n-1:]
        k1_freq = self.get_freq_method(k1, depth+1)
        k2_freq = self.get_freq_method(k2, depth+1)

        for i in range(2**tar_n):
            start = (i%(2**src_n))*2
            for j in range(start, start+2):
                k2_target_idxs = list(range(start, start+2, 1))
                prob = (k1_freq[i] / self._sum_all(k1_freq)) * (k2_freq[j] / self._sum_elements(k2_freq, k2_target_idxs)) / 2
                result.append(prob)
        return result

    def unigram_spacing(self, key):
        unigram_freq = self.get_freq_method(key)
        total_count = self._sum_all(unigram_freq)
        if total_count >= 1:
            return unigram_freq[3] / total_count
        return 0

    def get_bigram_prob(self, key:str, mode:int) -> float:
        result = self.min_prob
        bigram_freq = self.get_freq_method(key)
        total_count = self._sum_all(bigram_freq)
        
        if total_count>=1:
            target_idxs = self.bi_target_idx_dict[mode]
            result = self._sum_elements(bigram_freq, target_idxs) / total_count
        return result

    def _get_target_idxs(self, label:str, mode:int)->list:
        target_idxs = []
        new_label = label[:mode] + "1" + label[mode+1:]

        for i in range(len(new_label)):
            if new_label[i] == 0:
                target = int("0b" + new_label, 2)
                target_idxs.append(target)
            tmp_label = new_label[:i] + "1" + new_label[i+1:]
            target = int("0b" + tmp_label, 2)
            target_idxs.append(target)

        target_idxs = list(set(target_idxs))
        return target_idxs
    
    def _get_target_idxs_from_label(self, labels:str) -> list:
        targets = [""]
        for l in labels:
            tmp_targets = []
            for target in targets:
                if l!="2":
                    tmp_targets.append(target+l)
                else:
                    tmp_targets.append(target+"0")
                    tmp_targets.append(target+"1")
            targets = tmp_targets
        
        target_idxs = [int("0b"+target, 2) for target in targets]
        return target_idxs
        
    def get_bigram_prob_with_label(self, key:str, label:str, mode:int) -> float:
        if label[mode] == "1":
            return 1

        result = self.min_prob
        bigram_freq = self.get_freq_method(key)
        total_count = self._sum_all(bigram_freq)
        
        if total_count>=self.min_prob:
            target_idxs = self._get_target_idxs(label, mode)
            result = self._sum_elements(bigram_freq, target_idxs) / total_count

        return result

    def get_prob_with_idxs(self, key:str, idxs:list) -> float:
        result = self.min_prob
        freq = self.get_freq_method(key)
        total_count = self._sum_all(freq)

        if total_count >= self.min_prob:
            result = self._sum_elements(freq, idxs) / total_count

        return result


    def bigram_prob_for_tune(self, line:str) -> list:
        probability = []
        for i in range(1, len(line)-2):
            right = self.get_bigram_prob(line[i-1:i+1], mode=2)
            left = self.get_bigram_prob(line[i+1:i+3], mode=0)
            middle = self.get_bigram_prob(line[i:i+2], mode=1)
        
            probability.append((round(right, 4), round(middle, 4), round(left, 4)))
        return probability

    def step1_bigram_spacing(self, line:str, label:str, idx:int) -> list:
        key, val = line[idx-2:idx], label[idx-2:idx+1]
        target_idxs = self._get_target_idxs_from_label(val)
        right = self.get_prob_with_idxs(key, target_idxs)
        
        key, val = line[idx:idx+2], label[idx:idx+3]
        target_idxs = self._get_target_idxs_from_label(val)
        left = self.get_prob_with_idxs(key, target_idxs)

        key, val = line[idx-1:idx+1], label[idx-1:idx+2]
        target_idxs = self._get_target_idxs_from_label(val)
        middle = self.get_prob_with_idxs(key, target_idxs)

        total_weight, prob = 0, 0
        for w, p in zip([self.b1, self.b2, self.b3], [right, middle, left]):
            if p > self.min_prob:
                total_weight += w
                prob = prob + (w*p)
        #prob = self.b1*right + self.b2*middle + self.b3*left
        if total_weight>0:
            prob = prob/total_weight
        else:
            prob = 0.3
    
        return prob


    def step2_bigram_spacing(self, line:str, label:str) -> list:
        probability = []
        for i in range(1, len(line)-2):
            right = self.get_bigram_prob_with_label(line[i-1:i+1], label[i-1:i+2], mode=2)
            left = self.get_bigram_prob_with_label(line[i+1:i+3], label[i+1:i+4], mode=0)
            middle = self.get_bigram_prob_with_label(line[i:i+2], label[i:i+3], mode=1)

            prob = self.b1*right + self.b2*middle + self.b3*left
            probability.append(prob)
        return probability
    
    def get_trigram_prob(self, key:str, mode:int) -> float:
        result = self.min_prob
        trigram_freq = self.get_freq_method(key)
        total_count = self._sum_all(trigram_freq)
        
        if total_count>=1:
            target_idxs = self.tri_target_idx_dict[mode]
            result = self._sum_elements(trigram_freq, target_idxs) / total_count
        return result

    def get_trigram_prob_with_label(self, key:str, label:str, mode:int) -> float:
        if label[mode] == "1":
            return 1

        result = self.min_prob
        trigram_freq = self.get_freq_method(key)
        total_count = self._sum_all(trigram_freq)

        if total_count>=1:
            target_idxs = self._get_target_idxs(label, mode)
            result = self._sum_elements(trigram_freq, target_idxs) / total_count

        return result

    def trigram_prob_for_tune(self, line:str) -> list:
        probability = []
        for i in range(1, len(line)-2):
            if i == 1:
                right = self.get_prob_with_idxs(line[:2], idxs=[5,7])
                mid_right = self.get_trigram_prob(line[:3], mode=2)
            else:
                right = self.get_trigram_prob(line[i-2:i+1], mode=3)
                mid_right = self.get_trigram_prob(line[i-1:i+2], mode=2)
            
            if i == (len(line)-3):
                left = self.get_prob_with_idxs(line[i+1:i+3], idxs=[5, 7])
                mid_left = self.get_trigram_prob(line[i:i+3], mode=1)
            else:
                left = self.get_trigram_prob(line[i+1:i+4], mode=0)
                mid_left = self.get_trigram_prob(line[i:i+3], mode=1)
            
            probability.append((right, mid_right, mid_left, left))
        return probability

    def step1_trigram_spacing(self, line:str, label:str, idx:int) -> list:
        if idx == 2:
            right = self.get_prob_with_idxs(line[:2], idxs=[5,7])
        else:
            key, val = line[idx-3:idx], label[idx-3:idx+1]
            target_idxs = self._get_target_idxs_from_label(val)
            right = self.get_prob_with_idxs(key, target_idxs)

        key, val = line[idx-2:idx+1], label[idx-2:idx+2]
        target_idxs = self._get_target_idxs_from_label(val)
        mid_right = self.get_prob_with_idxs(key, target_idxs)

        if idx == (len(line)-3):
            left = self.get_prob_with_idxs(line[idx:idx+2], idxs=[5, 7])
        else:
            key, val = line[idx:idx+3], label[idx:idx+4]
            target_idxs = self._get_target_idxs_from_label(val)
            left = self.get_prob_with_idxs(key, target_idxs)

        key, val = line[idx-1:idx+2], label[idx-1:idx+3]
        target_idxs = self._get_target_idxs_from_label(val)
        mid_left = self.get_prob_with_idxs(key, target_idxs)

        total_weight, prob = 0, 0
        for w, p in zip([self.t1, self.t2, self.t3, self.t4], [right, mid_right, mid_left, left]):
            if p > self.min_prob:
                total_weight += w
                prob = prob + (w*p)
        #prob = self.t1*right + self.t2*mid_right + self.t3*mid_left + self.t4*left
        if total_weight>0:
            prob = prob/total_weight
        else:
            prob = 0.3
        return prob

    def step2_trigram_spacing(self, line:str, label:str) -> list:
        probability = []
        for i in range(1, len(line)-2):
            if i==1:
                right = self.get_prob_with_idxs(line[:2], idxs=[5, 7])
            else:
                right = self.get_trigram_prob_with_label(line[i-2:i+1], label[i-2:i+2], mode=3)
            mid_right = self.get_trigram_prob_with_label(line[i-1:i+2], label[i-1:i+3], mode=2)

            if i == (len(line)-3):
                left = self.get_prob_with_idxs(line[i+1:i+3], idxs=[5, 7])
            else:
                left = self.get_trigram_prob_with_label(line[i+1:i+4], label[i+1:i+5], mode=0)
            mid_left = self.get_trigram_prob_with_label(line[i:i+3], label[i:i+4], mode=1)
            
            prob = self.t1*right + self.t2*mid_right + self.t3*mid_left + self.t4*left
            probability.append(prob)
        return probability

    def pasting(self, sent:str):
        new_sent = ""
        sent_wo_space = sent.replace(" ", "")

        label = "1" + self.labeling(sent) + "11"
        sent_wo_space = self.start + sent_wo_space + self.end

        label = "".join([l if l=="1" else "2" for l in label])
        corrected_indexs = [0, len(label)]

        if " " not in sent:
            return sent

        index = 0
        prev_eojeol_len = 1
        eojeols = sent.split()
        for eojeol in eojeols[:-1]:
            threshold = self.pasting_threshold
            if (prev_eojeol_len + len(eojeol)<4):
                threshold = self.weak_pasting_threshold
            
            index += len(eojeol)
            if index < (len(sent_wo_space)-1):
                try:
                    bigram_p = self.step1_bigram_spacing(sent_wo_space, label, index)
                    trigram_p = self.step1_trigram_spacing(sent_wo_space, label, index)
                except:
                    bigram_p, trigram_p = 0.5, 0.5
                prob, total_weight = 0, 0

                for w, p in zip([self.w1, self.w2], [trigram_p, bigram_p]):
                    if p>0:
                        total_weight += w
                        prob += (w*p)

                if total_weight>0:
                    prob = prob/total_weight
                else:
                    prob = 1

                corrected_indexs.append(index)

                new_sent += eojeol
                prev_eojeol_len = len(eojeol)
                if prob>=threshold:
                    new_sent += " "

        new_sent += eojeols[-1]
        return new_sent.strip()

    def spacing(self, sent, corrected_indexs):
        new_sent = ""
        sent_wo_sapce = sent.replace(" ", "")

        label = self.labeling(sent) + "1"
        label = [l if i in corrected_indexs else "2" for i, l in enumerate(label)]
        label = "".join(label)

        for idx, eomjeol in enumerate(sent_wo_sapce):
            if (idx-1) in corrected_indexs:
                new_sent += eomjeol
                

    def spacing2(self, sent, trigram_p, bigram_p):
        new_sent = ""
        eojeols = sent.split()
        sent_wo_space = sent.replace(" ", "")

        index = 0
        for eojeol in eojeols:
            threshold = self.spacing_threshold
            if len(eojeol)>=5:
                threshold = self.weak_spacing_threshold
            
            i = index
            for i in range(index, index+len(eojeol)-1):
                prob = self.w1*trigram_p[i] + self.w2*bigram_p[i]
                new_sent += sent_wo_space[i]
                if prob > threshold:
                    new_sent += " "
            
            if i<(len(sent_wo_space)):
                new_sent = new_sent + eojeol[-1] + " "
            index += len(eojeol)
        
        return new_sent.strip()

    def correction(self, sent):
        sent = sent.replace(" ", " ")
        pattern_dict, prep_sen = util.prep(sent)
        new_sent = self.pasting(prep_sen)
        label = "1" + self.labeling(new_sent) + "11"
        sent_wo_space = self.start + new_sent.replace(" ", "") + self.end

        trigram_p = self.step2_trigram_spacing(sent_wo_space, label)
        bigram_p = self.step2_bigram_spacing(sent_wo_space, label)
        corrected = self.spacing2(new_sent, trigram_p, bigram_p)

        corrected = util.postprocessing(corrected, pattern_dict)

        return corrected