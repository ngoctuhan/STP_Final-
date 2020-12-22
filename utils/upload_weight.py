import pandas as pd 
import os
import numpy as np
from utils.params import Params

def __init_weight(prs: Params):

    # check  weight exsist
    if os.path.exists(prs.weight_file) == True:
        df = pd.read_csv(prs.weight_file, encoding='utf-8').values
        return df[:, 1:]
    else:

        num = len(prs.itent)
        weight = np.zeros((num, num))
        tmp = np.array(prs.itent).reshape(-1, 1)
        tmp2 = np.array(weight)
        merge = np.concatenate((tmp, tmp2), axis = 1)
        cols = [''] + prs.itent
        df = pd.DataFrame(merge,columns=cols)
        df.to_csv(prs.weight_file, index = False, encoding = 'utf-8')

        return weight


def update(bf, af, weight, prs:Params):

    if bf == 'non':
        return weight
    

    bf_enc =  prs.itent2num[bf]
    af_enc = prs.itent2num[af]
    weight[bf_enc][af_enc] += 1
    
    tmp = np.array(prs.itent).reshape(-1, 1)
    tmp2 = np.array(weight)
    cols = [''] + prs.itent
    merge = np.concatenate((tmp, tmp2), axis = 1)

    df = pd.DataFrame(merge,columns=cols)
    df.to_csv(prs.weight_file, index = False, encoding = 'utf-8')

    del tmp
    del tmp2
    del df
    return weight    


def get_top_intent(intent,weight ,prs:Params):

    
    id = prs.itent2num[intent]

    weight_row = weight[id]
    idx_max = np.argsort(weight_row)
   
    rel_itent = [prs.num2itent[i] for i in idx_max[-3:] if weight[id][i] > 0]

    return rel_itent


if __name__ == "__main__":

    prs = Params()

    w = __init_weight(prs)

    bf = 'submitCV'
    af = 'where'

    w = update(bf,af, w, prs)
    print(w)

    print(get_top_intent('submitCV', w ,prs))


    
    






