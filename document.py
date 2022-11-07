from re import findall
from json import dumps

class Document():
    def __init__(self, path):
        self.path = path
        self.doc_id = int(findall(r'\d+', self.path)[0])
        self.tf = self.create_tf()


    def create_tf(self):
        # open document file
        with open(self.path, 'r', encoding='UTF-8') as d:
            # get all terms while checking for blanks and new lines
            terms = d.read().strip().split()

        # create term frequency dictionarys
        tf = {}
        for term in terms:
            if term in tf:
                tf[term] += 1
            else:
                tf[term] = 1

        return tf


class Corpus():
    def __init__(self, documents):
        self.documents = documents
        self.inverted_index = {}

    
    def create_inverted_index(self):
        for d, id in self.documents:
            for key, value in d.items():
                if key in self.inverted_index:
                    self.inverted_index[key].append([id, value])
                else:
                    self.inverted_index[key] = [[id, value]]

        return self.inverted_index

    
    def save_inverted_index(self):
        with open('inverted_index.txt', 'w') as inv_ind:
            inv_ind.write(dumps(self.inverted_index))
