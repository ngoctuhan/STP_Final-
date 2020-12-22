import pickle
import time
from src.lm.suggestor import Suggestion
import random
import json
from src.ner.ner_extract import NER
from utils.covertPDF2TXT import pdf2txt
from utils.covertPDF2TXT import docx2txt

# import os
# path = 'dataset/ner/demo data'

# for filename in os.listdir(path):

#     print('filename: ', filename)
#     filepath = os.path.join(path, filename)
#     txt = pdf2txt(filepath, "outputF.txt")

#     ner = NER()
#     ner.predict(txt)
# inputF = "dataset/ner/demo data/sample_input.pdf"
# outputF = "dataset/ner/sample_input.txt"
# t0 = time.time()
# txt = pdf2txt(inputF, outputF)
# print(txt)

# ner = NER()
# t = time.time()
# ner.predict(txt)
# t2 = time.time()
# print(t-t0)
# # name_cv = 1
# # filename = 'dataset/ner/testdata.json'
# # with open(filename, 'r', encoding='utf-8') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         data = json.loads(line)
# #         text = data['content']
# #         save_name = 'cv_' + str(name_cv)+".txt"
# #         name_cv += 1
# #         with open(save_name, 'w', encoding='utf-8') as fs:
# #             fs.write(text)

# from src.nlu.mapping import Selena
# bot = Selena('models/MLP_model_nlu.pickle')


# print(bot.details_extention('Online'))
# from utils.pre_accents import gen_accents_word


# from src.lm.add_accents import Accentor

# acc = Accentor('model_LM')


# sentence = "ngay hom qua"

# print(acc.add_accents(sentence))

# sequences = [([x], 0.0) for x in gen_accents_word("yêu")]

# print(sequences)
sg = Suggestion('model_LM', ver=2)

print(sg.find_next_word(['ngày', 'hôm', 'nay']))


def load_obj(name, folder):
    with open(folder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


sg = Suggestion('model_LM', ver=2)

t1 = time.time()
res = sg.find_next_word(['em', 'muốn'])
print('Next word : em muốn', )
for word in res:
    print(word[0])
t2 = time.time()
print(t2-t1)

# sg.logscore(['nhân', 'dân'], 'sẽ')
# sg.logscore(['nhân', 'viên'], 'sẽ')
# sg.logscore(['nhân', 'dân'], 'mơ')
# sentence = []
# SENTENCE_LENGTH = 10
# vocab = load_obj('vocabv2', 'model_LM')
# unigram = load_obj('unigram', 'model_LM')
# total_words = unigram['']
# print(len(vocab))

# # randomize the first word
# rand_index = random.randint(0, total_words-1)
# first_word = list(vocab)[rand_index]
# sentence.append(first_word)

# print(sentence)
# for _ in range(SENTENCE_LENGTH-1):
#     word = sg.find_next_word(sentence)
#     sentence.append(word[0])
#     print(sentence)
# print(" ".join(sentence))


# total_words =


# sentence = []

# SENTENCE_LENGTH = 20

# rand_index = random.randint(0, total_words - 1)
# first_word = list()
