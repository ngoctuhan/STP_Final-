import numpy as np
import os
from underthesea import word_tokenize


class Processor_Text:

    def __init__(self, stop_word_file):

        self.stop_word_file = stop_word_file
        self.__load_stop_word()

    def __load_stop_word(self):

        self.stop_words = []
        file = open(self.stop_word_file, 'r', encoding='utf-8')
        for word in file:
            word = word.split("\n")[0]
            word = word.lower()
            self.stop_words.append(word)

        self.stop_words = np.unique(np.array(self.stop_words)).tolist()
        # print(self.stop_words)

    def remove_stop_word(self, sentence: list):

        new_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in self.stop_words:
                new_sentence.append(word)

        return new_sentence

    def synonyymous(self, words):

        new_words = []
        list_special = ['/', '[', ']', '', '+', '=', '?',
                        '.', ' ', ',', '-', '_', '(', ')', ':', '–']
        for word in words:
            if word.isnumeric() == True:
                word = 'number'
            if word in list_special:
                continue

            # check data format
            if len(word.split('/')) >= 2:
                word = 'date'
            new_words.append(word)
        words = new_words
        new_words = []
        for file in os.listdir('dataset/synonymous'):

            file_path = os.path.join('dataset/synonymous', file)
            f = open(file_path, 'r', encoding='utf-8').readlines()
            lines = [line.split('\n')[0] for line in f]
            # print(lines)
            for word in words:
                if word in lines:
                    word = file.split('.')[0]
                new_words.append(word)
            words = new_words
            new_words = []

        # words =  np.unique(np.array(words)).tolist()

        sentence = ""
        for i, word in enumerate(words):
            word = word.replace(" ", "_")
            if i == len(words) - 1:
                sentence = sentence + word
            else:
                sentence = sentence + word + " "
        return sentence

    def normalize_sentence(self, sentence):

        x = word_tokenize(sentence)
        x = self.remove_stop_word(x)
        sentence_new = self.synonyymous(x)
        return sentence_new
