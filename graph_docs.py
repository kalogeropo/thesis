from networkx import Graph
from document import Document

from numpy import array, transpose, dot, diagonal, fill_diagonal


class GraphDoc(Document):
    def __init__(self, path):
        super().__init__(path)
        self.adj_matrix = self.create_adj_matrix() 
        self.graph = None


    ################################################################################################
    # For each node we calculate the sum of the elements of the respective row or colum of its index#
    # as its degree                                                                                 
    ################################################################################################
    # For more info see LEMMA 1 and LEMMA 2 of P: A graph based extension for the Set-Based Model, A: Doukas-Makris
    def create_adj_matrix(self):
        # get list of term frequencies
        rows = array(list(self.tf.values()))

        # reshape list to column and row vector
        row = transpose(rows.reshape(1, rows.shape[0]))
        col = transpose(rows.reshape(rows.shape[0], 1))
        
        # create adjecency matrix by dot product
        adj_matrix = dot(row, col)

        # calculate Win weights (diagonal terms)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i == j:
                    adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5 # Win
        
        return adj_matrix


    def create_graph_from_adjmatrix(self, **kwargs):
    
        filename = kwargs.get('filename', None)
        if not filename:
            filename = 'Name not found!' # used when i want to visualize graphs with name

        # check if adj matrix not built yet
        if self.adj_matrix is None:
            self.adj_matrix = self.create_adj_matrix()
    
        self.graph = Graph()
        termlist = list(self.tf.keys())
        for i in range(self.adj_matrix.shape[0]):
            self.graph.add_node(i, term=termlist[i])
            for j in range(self.adj_matrix.shape[1]):
                if i > j:
                    self.graph.add_edge(i, j, weight=self.adj_matrix[i][j])
        # graphToPng(gr,filename = filename)

        return self


    def uniongraph(terms, term_freq, adjmatrix, collection_terms, union_graph_termlist_id, union_gr, id,
               collection_term_freq, kcore, kcorebool):
        #auto douleuei dioti o adjacency matrix proerxetai apo ton admatrix tou pruned graph opws kai o kcore ara uparxei tautisi twn diktwn (i) twn kombwn
        for i in range(adjmatrix.shape[0]):
            h = 0.06 if i in kcore and kcorebool else 1
            
            if terms[i] not in collection_terms:
                collection_terms[terms[i]] = id
                union_graph_termlist_id.append(id)
                collection_term_freq.append(term_freq[i] * (term_freq[i] + 1) * 0.5 * 0.05)
                union_gr.add_node(terms[i], id=id)
                id += 1
            elif terms[i] in collection_terms:
                index = collection_terms[terms[i]]
                collection_term_freq[index] += term_freq[i] * (term_freq[i] + 1) * 0.5 * 0.05
            for j in range(adjmatrix.shape[1]):
                if i > j:
                    if adjmatrix[i][j] != 0:
                        if terms[j] not in collection_terms:
                            collection_terms[terms[j]] = id
                            union_graph_termlist_id.append(id)
                            collection_term_freq.append(term_freq[j] * (term_freq[j] + 1) * 0.5 * 0.05)
                            union_gr.add_node(terms[i], id=id)
                            id += 1
                        # print('kcorbool = %s and h = %d '%(str(kcorebool),h ))
                        if not union_gr.has_edge(terms[i], terms[j]):
                            union_gr.add_edge(terms[i], terms[j], weight=adjmatrix[i][j] * 0.05)
                            #print("Calculating adj[",i,"]",j,"]: ",adjmatrix[i][j] * h)
                        elif union_gr.has_edge(terms[i], terms[j]):
                            union_gr[terms[i]][terms[j]]['weight'] += adjmatrix[i][j] * 0.05
                            #print("Calculating adj[",i,"]",j,"]: ",adjmatrix[i][j] * h)

        return terms, adjmatrix, collection_terms, union_graph_termlist_id, union_gr, id, collection_term_freq
