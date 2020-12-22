import time
import spacy
from src.ner.train import convert_dataturks_to_spacy


class NER:

    def __init__(self):
        self.nlp_name = spacy.load('model_NER/Name_Email')
        self.nlp_college = spacy.load('model_NER/College')
        self.nlp_designation = spacy.load('model_NER/Designation')
        self.nlp_skills = spacy.load('model_NER/Skills')
        self.nlp_work = spacy.load('model_NER/Work')

    def predict(self, text):

        name_email = self.nlp_name(text)
        d = {}
        for ent in name_email.ents:
            d[ent.label_] = []
        for ent in name_email.ents:
            d[ent.label_].append(ent.text)

        # .................................
        college = self.nlp_college(text)
        for ent in college.ents:
            d[ent.label_] = []
        for ent in college.ents:
            d[ent.label_].append(ent.text)

        # .....................................

        des = self.nlp_designation(text)
        for ent in des.ents:
            d[ent.label_] = []
        for ent in des.ents:
            d[ent.label_].append(ent.text)

        # .....................................

        skills = self.nlp_skills(text)
        for ent in skills.ents:
            d[ent.label_] = []
        for ent in skills.ents:
            d[ent.label_].append(ent.text)

        # .....................................

        work = self.nlp_work(text)
        for ent in work.ents:
            d[ent.label_] = []
        for ent in work.ents:
            d[ent.label_].append(ent.text)

        print(d)
        return d


if __name__ == "__main__":

    text = "Abhishek Jha\nApplication Development Associate - Accenture\n\nBengaluru, Karnataka - Email me on Indeed: indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a\n\n• To work for an organization which provides me the opportunity to improve my skills\nand knowledge for my individual and company's growth in best possible ways.\n\nWilling to relocate to: Bangalore, Karnataka\n\nWORK EXPERIENCE\n\nApplication Development Associate\n\nAccenture -\n\nNovember 2017 to Present\n\nRole: Currently working on Chat-bot. Developing Backend Oracle PeopleSoft Queries\nfor the Bot which will be triggered based on given input. Also, Training the bot for different possible\nutterances(Both positive and negative), which will be given as\ninput by the user.\n\nEDUCATION\n\nB.E in Information science and engineering\n\nB.v.b college of engineering and technology - Hubli, Karnataka\n\nAugust 2013 to June 2017\n\n12th in Mathematics\n\nWoodbine modern school\n\nApril 2011 to March 2013\n\n10th\n\nKendriya Vidyalaya\n\nApril 2001 to March 2011\n\nSKILLS\n\nC(Less than 1 year), Database(Less than 1 year), Database Management(Less than 1 year), \nDatabase Management System(Less than 1 year), Java(Less than 1 year)\n\nADDITIONAL INFORMATION\n\nTechnical Skills\n\nhttps: // www.indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a?isid = rex-download & ikw = download-top & co = IN\n\n\n• Programming language: C, C++, Java\n• Oracle PeopleSoft\n• Internet Of Things\n• Machine Learning\n• Database Management System\n• Computer Networks\n• Operating System worked on: Linux, Windows, Mac\n\nNon - Technical Skills\n\n• Honest and Hard-Working\n• Tolerant and Flexible to Different Situations\n• Polite and Calm\n• Team-Player"

    ner = NER()

    ner.predict(text)
