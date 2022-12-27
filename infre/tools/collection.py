from json import dumps, load
from os.path import exists, join
from os import makedirs, listdir, getcwd
from gensim.utils import simple_preprocess

from infre.tools.document import Document
from networkx import to_numpy_array, is_empty
from numpy import fill_diagonal

# collection would be either load from path or create from path
# load from path -> col_path
# create from path -> path
# load variable will determine if Collection obj will make indexes from scrath or load them from existing path
# graph_docs will be optional for other uses

class Collection():
    def __init__(self, path, docs=[]):

        self.path = join(getcwd(), path)

        self.number = len(listdir(self.path))
        print(self.number)

        # can be used to hold different user given information
        self.params = {}

        # Class Representation of each document
        self.docs = docs

        # inverted index 
        self.inverted_index = {}


    def create_collection(self):
        
        if not self.docs:
            self.docs = self.create_documents()
        self.inverted_index = self.create_inverted_index()

        return self    


    def create_documents(self):
    
        # list files
        filenames = [join(self.path, f) for f in listdir(self.path)]

        # list to hold every Document obj
        docs = []
        # for every document file
        for filename in filenames: 
            docs += [Document(filename)]

        return docs


    def create_inverted_index(self):

        # nwk = self.calculate_nwk()
        id, cnt = 0, 0
        inv_index = {}

        for doc in self.docs:
            for term, tf in doc.tf.items():
                try:
                    if term not in inv_index:
                        inv_index[term] = {
                                            'id': id,
                                            'total_tf': tf,
                                            'posting_list': [[doc.doc_id, tf]],
                                            # 'nwk': nwk[key],
                                            'term': term
                                        }
                        id += 1
                    elif term in inv_index:
                        inv_index[term]['total_tf'] += tf
                        inv_index[term]['posting_list'] += [[doc.doc_id, tf]]
                except KeyError:
                    cnt += 1
                    print(f"Keys not found {cnt}")

        return inv_index


    def get_inverted_index(self):
        return self.inverted_index


    def create_directory(self):
        # check if exists else create directories
        for path in self.path.values(): 
            if not exists(path): makedirs(path)


    ###### NEEDS REMODELLING ##########
    def save_inverted_index(self, name='inv_index.json'):
        # define indexes path
        path = join(self.path['index_path'], name)

        try: 
            with open(path, 'w', encoding='UTF-8') as inv_ind:
                # create inv ind if not created 
                if not self.inverted_index:
                    self.inverted_index()
                # store as JSON
                inv_ind.write(dumps(self.inverted_index))

         # if directory does not exist
        except FileNotFoundError:
                # create directory
                self.create_model_directory()
                # call method recursively to complete the job
                self.save_inverted_index()
        finally: # if fails again, reteurn object
            return self


    ###### NEEDS REMODELLING ##########
    def load_inverted_index(self, name="inv_index.json"):

        # path to find stored graph index
        path = join(self.path['index_path'], name)

        try:
            # open file and read as dict while reconstructing the data as a dictionary
            with open(path) as f: self.inverted_index = load(f)

        except FileNotFoundError:
            raise('There is no such file to load collection.')

        return self.inverted_index


    def get_adj_matrix(self):
        
        if is_empty(self.graph):
            try:
                self.load_graph()
            except:
                self.graph = self.union_graph()

        adj = to_numpy_array(self.graph)
        adj_diagonal = list(self.calculate_win().values())
        fill_diagonal(adj, adj_diagonal)
        
        return adj


    def preprocess(self, document_terms):
        from gensim.utils import simple_preprocess
        from nltk.corpus import stopwords
        # from nltk.stem import WordNetLemmatizer
        punc_free_terms = simple_preprocess(' '.join(term for term in document_terms), min_len=1, max_len=30)
        
        stop_words = stopwords.words('english')
        filtered_words = [term for term in punc_free_terms if term not in stop_words]
        
        #defining the object for Lemmatization
        # wordnet_lemmatizer = WordNetLemmatizer()
        # lemm_terms = [wordnet_lemmatizer.lemmatize(term) for term in filtered_words]
    
        return filtered_words


    def load_qd(self):
       
        with open(join('collections/CF', 'Queries.txt'), 'r') as fd:
            queries = [self.preprocess(q.split()) for q in fd.readlines()]
            print(queries[0:10])
        with open(join('collections/CF', 'Relevant.txt'), 'r') as fd:
            relevant = [[int(id) for id in d.split()] for d in fd.readlines()]

        return queries, relevant
