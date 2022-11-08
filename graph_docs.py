from networkx import Graph, disjoint_union_all
from document import Document


from numpy import array, transpose, dot, diagonal, fill_diagonal


class GraphDoc(Document):
    def __init__(self, path):
        super().__init__(path)
        self.adj_matrix = self.create_adj_matrix() 
        self.graph = self.create_graph_from_adjmatrix()


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
    
        graph = Graph()
        termlist = list(self.tf.keys())
        for i in range(self.adj_matrix.shape[0]):
            graph.add_node(i, term=termlist[i])
            for j in range(self.adj_matrix.shape[1]):
                if i > j:
                    graph.add_edge(i, j, weight=self.adj_matrix[i][j])
        # self.graphToPng(gr,filename = filename)

        return graph


class GraphUnion():
    def __init__(self, graph_docs):
        self.graph_docs = graph_docs


    def union_graph(self, kcore=[], kcorebool=False):
        # empty union at first
        union_graph = Graph()
        
        for gd in self.graph_docs:
            adj_matrix = gd.adj_matrix
            terms = list(gd.tf.keys())
            fq = list(gd.tf.values())

            for i in range(adj_matrix.shape[0]):
                h = 0.06 if i in kcore and kcorebool == True else 1
                
                w_in = fq[i] * (fq[i] + 1) * 0.5 * h
                if not union_graph.has_node(terms[i]):
                    union_graph.add_node(terms[i], weight=w_in)
                # else re-weight
                elif union_graph.has_node(terms[i]):
                    union_graph.nodes[terms[i]]['weight'] += w_in
                
                # visit only lower diagonal
                for j in range(adj_matrix.shape[1]):
                    if i > j:
                        w_in_ng = fq[j] * (fq[j] + 1) * 0.5 * h
                        if not union_graph.has_edge(terms[i], terms[j]):
                            # node(term[j] auto created)
                            # assign Win weight
                            union_graph.nodes[terms[j]]['weight'] = w_in_ng
                            # assign Wout weight
                            union_graph.add_edge(terms[i], terms[j], weight=adj_matrix[i][j] * h)
                        elif union_graph.has_edge(terms[i], terms[j]):
                            union_graph[terms[i]][terms[j]]['weight'] += adj_matrix[i][j] * h

            return union_graph

