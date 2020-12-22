import os
import numpy as np 
import gensim
import time

class Encoder_Sentence:

    def __init__(self, model_path):

        self.model_path = model_path
        self.__init_Model()

    def __init_Model(self):

        print("[INFOR]: Loading pre-train model Word2Vector from disk to RAM....")
        t1 = time.time()
        self.model = gensim.models.Word2Vec.load(self.model_path)
        t2 = time.time()
        print(">>>>>[INFOR]: Load done after {} !".format(str(t2-t1)))

    def embeding(self, sentence):

        """
        Encoder sentence to vector using Word2vec model 
        Output_dim = 300
        """

        pass


end = Encoder_Sentence('word2vec/wiki_v2.vi.vec')