import re
import copy

english = re.compile('[a-zA-Z]+')
number = re.compile('[0-9]+')
hanja = re.compile('[一-龥]+')

to_eng = re.compile('Z')
to_num = re.compile('0')
to_hanja = re.compile('演')

mode_to_func = {
    "eng" : to_eng,
    "num" : to_num,
    "hanja" : to_hanja
}

def prep(sentence):
    filter_dict = {
        "eng":english.findall(sentence),
        "num":number.findall(sentence),
        "hanja":hanja.findall(sentence),
    }
    prep_sen = english.sub("Z", sentence)
    prep_sen = number.sub("0", prep_sen)
    prep_sen = hanja.sub("演", prep_sen)

    return filter_dict, prep_sen

def rematching(sen, filter_dict, mode):
    shift = 0
    after_sen = copy.deepcopy(sen)
    
    for i, x in enumerate(mode_to_func[mode].finditer(sen)):
        org_token = filter_dict[mode][i]
        start = x.span()[0] + shift
        after_sen = after_sen[:start] + org_token + after_sen[start+1:]
        shift += (len(org_token)-1)
    
    return after_sen


def postprocessing(corrected, filter_dict):
    for key in mode_to_func.keys():
        corrected = rematching(corrected, filter_dict, key)
    return corrected