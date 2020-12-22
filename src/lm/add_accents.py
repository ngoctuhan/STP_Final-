from src.lm.suggestor import Suggestion
from utils.pre_accents import gen_accents_word


class Accentor:
    def __init__(self, model_path):

        self.model_path = model_path
        self.model = Suggestion(self.model_path, ver=2)

    def beam_search(self, words, k=3):
        sequences = []
        for idx, word in enumerate(words):
            if idx == 0:
                sequences = [([x], 0.0) for x in gen_accents_word(word)]
            else:
                all_sequences = []
                for seq in sequences:

                    for next_word in gen_accents_word(word):
                        current_word = seq[0][-1]

                    try:
                        previous_word = seq[0][-2]
                        score = self.model.logscore(
                            next_word, [previous_word, current_word])
                    except:
                        score = self.model.logscore(next_word, [current_word])
                    new_seq = seq[0].copy()
                    new_seq.append(next_word)
                    all_sequences.append((new_seq, seq[1] + score))
                all_sequences = sorted(
                    all_sequences, key=lambda x: x[1], reverse=True)
                print(all_sequences)
                sequences = all_sequences[:k]
        return sequences

    def add_accents(self, sentence):

        # sentence = "ngay hom qua la ngay bau cư tong thong My"
        words = sentence.lower().split()
        self.beam_search(words)
