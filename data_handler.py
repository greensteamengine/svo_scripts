import spacy
import string
import csv
import torch
from os.path import exists

class CustomException(Exception):
    pass

class DataHandler:
    
    def __init__(self):
        #https://spacy.io/models/en
        self.nlp = spacy.load("en_core_web_sm")

    def simple_sentences_from_csv(self, file_path, limit=None):
        """
        Adopted for the sentences used were from https://www.kaggle.com/mfekadu/sentences:
        sentence number,sentnece
        e.g.
        11,Example sentence
        """
        sentences = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            for row in reader:
                sentences.append(row[1])
                count+=1
                if(limit != None and count >= limit):
                    break
        #the first one s an empy sentence
        return sentences[1:]
    
    #minaly from here, a bit moddified
    # https://github.com/Dimev/Spacy-SVO-extraction/blob/master/main.py
    
    # extract the subject, object and verb from the input
    # https://nlp.stanford.edu/software/dependencies_manual.pdf
    def extract_svo(self, sentence):
        
        #https://spacy.io/api/dependencyparser
        #obj: direct object
        #https://universaldependencies.org/u/dep/
        
        #best descriptions:
        #dependency labels: https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
        
        
        # https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf
        OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
        #nsubj: nominal subject <- https://universaldependencies.org/docs/en/dep/nsubj.html
        #nsubjpass: passive nominal subject <- https://universaldependencies.org/docs/en/dep/nsubjpass.html
        # csubj: clausal subject <- https://universaldependencies.org/docs/en/dep/csubj.html
        SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass" "agent", "expl"}
        
        sub, sub_i = [], []
        ob, ob_i = [], []
        ve, ve_i = [], []
        
        doc = self.nlp(sentence)
        
        tokens = []
        
        #https://spacy.io/api/token
        for token in doc:
            tokens.append(token.text)
            #print(token.pos)
            # is this a verb?
            #underscore means str format for Coarse-grained part-of-speech from the Universal POS tag set.
            # https://universaldependencies.org/u/pos/
            if token.pos_ == "VERB" or token.dep_ == "ROOT":
            
                ve.append(token.text)
                ve_i.append(token.i)
            # is this the object?
            if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
                ob.append(token.text)
                ob_i.append(token.i)
            # is this the subject?
            if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
                sub.append(token.text)
                sub_i.append(token.i)
            
    
        return [sentence, tokens, [sub_i, sub], [ve_i, ve], [ob_i, ob]]
    
    '''
    some sentence data manipulation functions insertede here, if the structure of the sentece data were changed,
    its enough to make changes here, and no changes will be necessery elsewhere
    '''
    def get_sentence(self, sentence_data):
        return sentence_data[0]
    
    def get_tokens(self, sentence_data):
        return sentence_data[1]
    
    def get_svo(self, sentence_data, part):
        if (part == 's'):
            return sentence_data[2]
        elif(part == 'v'):
            return sentence_data[3]
        elif(part == 'o'):
            return sentence_data[4]
        else:
            raise CustomException(f"Wrong part-of-speech indicated: >{part}<")

    #format:
    #['He was ac....ery', ['He', ..., 'Cemetery'], [[0], ['He']], [[2, 8], ['accorded', 'buried']], [[3, 4, 5], ['a', 'State', 'funeral']]]
    def from_str_sentence_data(self,  svo_attributes_str):
        
        parts = svo_attributes_str.split('#')
        stc = parts[0]
        tokens = parts[1].split('_')
        
        s_idx, s_related = [], []
        if parts[2]:
            s_idx = list(map(int, parts[2].split('_')))
            s_related = parts[3].split('_')
        
        v_idx, v_related = [], []
        if parts[4]:
            v_idx = list(map(int, parts[4].split('_')))
            v_related = parts[5].split('_')
        
        o_idx, o_related = [], []
        if parts[6]:
            o_idx = list(map(int, parts[6].split('_')))
            o_related = parts[7].split('_')
        
        
        return [stc, tokens, [s_idx, s_related], [v_idx, v_related], [o_idx, o_related]]
    
    
    def to_str_sentence_data(self,  svo):
        
        stc = svo[0]
        tokens = '_'.join(svo[1])
        
        s_idx = '_'.join(list(map(str, svo[2][0])))
        s_related = '_'.join(svo[2][1])
        
        v_idx = '_'.join(list(map(str, svo[3][0])))
        v_related = '_'.join(svo[3][1])
        
        o_idx = '_'.join(list(map(str, svo[4][0])))
        o_related = '_'.join(svo[4][1])
        
        final_str = '#'.join([stc, tokens, s_idx, s_related, v_idx, v_related, o_idx, o_related])
        
        return final_str
    
    
    
    
    def create_dataset(self, file_path, target_file_path, write_to_file=False, limit=None):
        

        sentences = self.simple_sentences_from_csv(file_path, limit)
        
        result_sentences = []

        for idx, s in enumerate(sentences):
            if(idx%1000 == 0):
                print(f"Creating data: {idx}/{len(sentences)}")
                
            svo_attributes = self.extract_svo(s)
            
            result_sentences.append(svo_attributes)
            
        if (write_to_file):
            with open(target_file_path, 'w') as outfile:
                for res_s in result_sentences:
                    outfile.write(f"{self.to_str_sentence_data(res_s)}\n")
                
        return result_sentences
    
    def read_dataset(self, file_path, limit):
        
        if (not exists(file_path)):
            print("Missing data file!\nPlease download from https://www.kaggle.com/mfekadu/sentences the file cv-unique-no-end-punct-sentences.csv")
        
        sentence_data = []
        with open(file_path) as file:
            lines = file.readlines()
            if limit is None:
                limit = len(lines)
            for l in lines[:limit]:
                snt = self.from_str_sentence_data(l[:-1])
                sentence_data.append(snt)
        return sentence_data

                


if __name__ == "__main__":

    
    nlp = spacy.load("en_core_web_sm")
    create_dataset("cv-unique-no-end-punct-sentences.csv")

    

    ex1 = "Over time it became available at various levels."

    doc = nlp(ex1)
    for token in doc:
        print(f"{token} pos: {token.pos_} dep: {token.dep_}")
    #print(nlp(example))
