import pickle


def load_obj(name, folder):
    with open(folder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Suggestion:
    def __init__(self, model_path, ver=1):

        self.model_path = model_path
        self.ver = ver
        if self.ver == 1:
            self.__load_model()
        else:
            self.__load_model_v2()

    def __load_model(self):

        self.unigram = load_obj('unigram', self.model_path)
        self.bigram = load_obj('bigram', self.model_path)
        self.trigram = load_obj('trigram', self.model_path)
        self.vocab = load_obj('vocal', self.model_path)

    def __load_model_v2(self):

        self.unigram = load_obj('unigramv2', self.model_path)
        self.bigram = load_obj('bigramv2', self.model_path)
        self.trigram = load_obj('trigramv2', self.model_path)
        self.vocab = load_obj('vocabv2', self.model_path)

    def get_prob_unigram(self, word):
        if word not in self.unigram:
            return 0
        return self.unigram[word] / self.unigram['']

    def get_prob_bigram(self, words):
        if words not in self.bigram:
            return 0
        return self.bigram[words] / self.unigram[words[0]]

    def get_prob_trigram(self, words):
        if words not in self.trigram:
            return 0
        return self.trigram[words] / self.bigram[words[:2]]

    def find_next_word(self, words):
        candidate_list = []

        # Loop through all words
        # Check for the probability if we were to generate this word
        for word in self.vocab:
            p1 = self.get_prob_unigram((word))
            p2 = self.get_prob_bigram((words[-1], word))
            p3 = self.get_prob_trigram(
                (words[-2], words[-1], word)) if len(words) >= 2 else 0

            # We use linear interpolation
            p = 0.01*p1 + 0.4*p2 + 0.5*p3

            candidate_list.append((word, p))

        # sort based on the score and select the best one
        candidate_list.sort(key=lambda x: x[1], reverse=True)
        return candidate_list[:10]

    def logscore(self, next_word, words):

        p1 = self.get_prob_unigram((next_word))
        p2 = self.get_prob_bigram((words[-1], next_word))
        p3 = self.get_prob_trigram(
            (words[-2], words[-1], next_word)) if len(words) >= 2 else 0

        p = 0.01*p1 + 0.4*p2 + 0.5*p3

        return p
