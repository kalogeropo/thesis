import os

import time

import sys
import csv

import string
import pandas as pd

from openpyxl import load_workbook

translator = str.maketrans('', '', string.punctuation)
# parsing a file to an inverted index and return a list of unique terms and term frequency

from k_core_modules import * #helper functs
from graph_creation_scripts import *
from graphs import *




def runIt(filename, ans,window_flag, window_size, sen_par_flag, par_window_size,per_window,dot_split,file_sum_mat):
    print(filename)
    temp = createInvertedIndexFromFile(filename, postinglist)
    # temp[0] = terms | temp[1] = term freq | temp[2] = posting list | temp 3 = doc number of words
    
    #######TEST############
    #print(temp[0])
    #print('\n')
    #print(temp[1])
    #print('\n')
    #print(temp[2])
    #print('\n')
    #print(temp[3])
    #print('\n')
    
    if window_flag:
        try:
            adjmat = CreateAdjMatrixFromInvIndexWithWindow(temp[0],filename,window_size,per_window,dot_split)
        except MemoryError:
            sizeof_err_matrix = sys.getsizeof(adjmat)
            print(sizeof_err_matrix)
            exit(-1)        
    elif sen_par_flag:
        try:
            adjmat = CreateAdjMatrixFromInvIndexWithSenParWindow(temp[0],filename,window_size, par_window_size,dot_split)
        except MemoryError:
            sizeof_err_matrix = sys.getsizeof(adjmat)
            print(sizeof_err_matrix)
            exit(-1) 
    else:
        try:
            adjmat = CreateAdjMatrixFromInvIndex(temp[0], temp[1])
        except MemoryError:
            sizeof_err_matrix = sys.getsizeof(adjmat)
            print(sizeof_err_matrix)
            exit(-1)
    #if int(filename[9:]) in bucket_list[5]:
     #   file_sum_mat = calculateSummationMatrix(adjmat,filename,file_sum_mat,temp[0],window_size)
    #######################
    try:
        gr = graphUsingAdjMatrix(adjmat, temp[0])
        docinfo.append([filename, temp[3]])
    except MemoryError:
        sizeof_err_matrix = sys.getsizeof(adjmat)
        print(sizeof_err_matrix)
        exit(-1)
    with open('docinfo.dat', 'a') as file_handler:
        file_handler.write('%s %s \n' % (filename, temp[3]))
    file_handler.close()
    # print("----------------Using networkx method:---------------")
    # calculate the difference between min and max similarity and use it to prune our graph
    kcore = nx.Graph()
    kcore_nodes = []
    prunedadjm = nx.to_numpy_array(kcore)
    #	corebool --> not used(Can change to apply union graph penalty or not)
    #	splitfiles --> Window based splitting methods will be used if true(Exists because we use GSB in our experiments which doesnt require splitting)
    #	sen_par_flag --> if true sentence paragraph method will be used
    #	dot_split --> if true we will split according to "." using nltk's tokenize with punkt
    #	window_size --> The size of window when we are splitting using constant windows. If it is equal to 0
    #			per_window will be used instead.(per_window is the percentage of the text we will use)
    #	per_window --> A number between 0-1. 0 is 0% of the text while 1 is 100%. It is used to calculate the
    #			window size when using file length percentage based splitting.
    #	par_window_size --> Only used when sen_par_flag is true. It is the window size that corresponds to the paragraph level
    #	invfilename --> filename of the inverted index that will be constructed.
    #
    # Flag Hierarchy:
    #
    #	splitfiles >> sen_par_flag >> dot_split >> corebool(doesnt do anyting yet)
    #
    #	if splitfiles is false then we use gsb
    #
    #	if sen_par_flag is true we use sentence/paragraph splitting
    #		if dot_split is true the sentence portion of the adjmatrix will be split according to "."
    #		if dot split is false the file will be split according to window size(the sentence part only)
    #			if window size is 0 then the sentence portion of the adjmatrix will be generated according to percentage of the file(based on per_window)
    #			if window size is a positive integer the sentence portion of the adjmatrix will be generated according to that integer(constant window splitting)
    #		For the paragraph part we will be using constant window splitting according to par_window_size.(always)
    #	if sen_par_flag is false then we use regular splitting accoding to the rest of the flags/values
    #		if dot_split is true then we use splitting according to "." using nltk and punkt
    #		if dot split is false the file will be split according to window size
    #			if window size is 0 then the file will be split according to percentage of the file(based on per_window)
    #			if window size is a positive integer the file will be split according to that integer(constant window splitting)
    #
    #
    #
    # With current flags:
    #
    #	X=1 --> penalty on union graph splitting using percentages
    #	X=2 --> GSB
    #	X=3 --> penalty on union graph splitting using percentages
    #	X=4 --> penalty on union graph splitting using "." ISSUE:IF nltk is not installed, BY PASS: At menu 2: input one of existing indexes
    #	X=5 --> penalty on union graph splitting using constant window size
    #	X=6 --> penalty on union graph splitting using sentence/paragraph windows
    #
    #   !This is where we add methods to improve the graph such as core/truss decomposition, pruning, methods for important nodes. !

    #   NOTE: this main uses penalty on union graph (see lines 65,70,77 - uniongraph function) to punish frequent edges.
    if ans == 1:

        # By creating new graph we can translate it easily to the respective adj matrix
        # without calculating each edge weight separtly. It returns a pruned GRAPH
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []
        #stopwordsStats(kcore,temp[0],filename)
        #print(nx.number_of_nodes(kcore))
        #print(len(kcore))
        #print(kcore.degree())
    if ans == 3:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes =[]
    if ans == 4 :
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes =[]
            
    ###################TEST####################
    if ans == 5 :
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes =[]
		
    if ans == 6 :
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes =[]
    
    #########################################################
    
    if ans == 2:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes =[]
    term_freq = temp[1]
    # print(term_freq)
    return adjmat, temp[0], gr, term_freq, kcore_nodes, prunedadjm, file_sum_mat  # adjacency matrix terms list , graph


#############################--PREPROCCESS THE FILES ------------######
print('===========Preproccess==========\n')
start = time.time()
file_list = []
for item in os.listdir('txtfiles'):
    name = os.path.join("txtfiles", item)
    # print(os.path.getsize(name))
    if os.path.getsize(name) == 0:
        print('%s is empty:' % name)
        os.remove(name)
    else:
        #preproccess(name) #possible bug here uncomment this line if the files are used for the first time
        file_list.append([name, os.path.getsize(name)])
file_list = sorted(file_list, key=itemgetter(1), reverse=True)
#print(file_list)
end = time.time()
print('--------->preproccess took %f mins \n' % ((end - start) / 60))
######################## HASHING #####################################
bucket_list = []
#bucket_list = BucketHash(file_list)
#print(bucket_list)
########################--------test-------------#####################


union_graph_termlist_id = []  # id of each unique word in the collection
id = 0  # index used to iterate  on every list
collection_terms = {}  # unique terms in the collection as a dict for performance
union_graph = nx.Graph()  # Union of each graph of every document
collection_term_freq = []
############################################
file_sum_mat = [] #file and summation matrix
stopword_weight_mat = []
############################################
sumtime = 0
'''prints the menu '''
menu = printmenu()
print(menu)
S = menu[2]
hargs=menu[1]
menu = menu[0]
#######TEST#######
if len(sys.argv)<= 8 and len(sys.argv)>=5:
    X = sys.argv[4]
elif len(sys.argv)!=4:
    print("some error here")
    exit(-99)
###################

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!if menu==1 implement
if menu == 1:
    #X = input(' 1.create index using maincore \n 2.create index without considering maincore \n 3.Create index using Density method \n 4.Create index using CoreRank method\n' )
    #

    if int(X) == 1:
        corebool = True
        splitfiles = True
        sen_par_flag = False
        dot_split = False
        window_size = 0
        par_window_size = 0
        per_window = float(sys.argv[7])
        invfilename = 'NegMain.dat'
    elif int(X) == 2:
        corebool = False
        splitfiles = False
        sen_par_flag = False
        dot_split = False
        window_size = 0
        par_window_size = 0
        per_window = 0
        invfilename = 'invertedindex.dat'
    elif int(X) == 3:
        corebool = False
        splitfiles = True
        sen_par_flag = False
        dot_split = False
        window_size = 0
        par_window_size = 0
        per_window = float(sys.argv[7])
        invfilename = 'PerSplit.dat'
    elif int(X) == 4:
        corebool = False
        splitfiles = True
        sen_par_flag = False
        dot_split = True
        window_size = 0
        par_window_size = 0
        per_window = 0
        invfilename = 'DotSplit.dat'
    ########TEST##############
    elif int(X) == 5:
        splitfiles = True
        sen_par_flag = False
        corebool = False
        dot_split = False
        par_window_size = 0
        window_size = int(sys.argv[5])
        per_window = 0
        invfilename = 'ConstantWindFile.dat'
    elif int(X) == 6:
        sen_par_flag = True
        splitfiles = False
        corebool = False
        dot_split = False
        window_size = int(sys.argv[5])
        par_window_size = int(sys.argv[6])
        per_window = 0
        invfilename = 'SenParConWind.dat'
    ##########################
    
    un_start = time.time()
    start = time.time()
    remaining = len(os.listdir('txtfiles')) + 1

    for name in file_list:
        remaining -= 1

        print(remaining)
        name = name[0]
        # print("=========================For file = %s==================== " % name)    
        gr = runIt(name, int(X),splitfiles, window_size, sen_par_flag, par_window_size,per_window,dot_split,file_sum_mat)
        # write doc info to file
        adjmatrix = gr[0]
        terms = gr[1]
        graph = gr[2]
        term_freq = gr[3]
        maincore = gr[4]
        prunedadjm = gr[5]
        file_sum_mat = gr[6]
        #getGraphStats(graph,name,True, True)

        try:
            ug = uniongraph(terms, term_freq, adjmatrix, collection_terms, union_graph_termlist_id, union_graph, id,
                            collection_term_freq, maincore, kcorebool=corebool)
            
        except MemoryError:
            sizeofgraph = sys.getsizeof(union_graph.edge) + sys.getsizeof(union_graph.node)
            print(sizeofgraph)
            exit(-1)
        #######################
        collection_terms = ug[2]
        #######################
        id = ug[5]
        union_graph = ug[4]
        collection_term_freq = ug[6]

        un_end = time.time()
        sumtime += (un_end - un_start) / 60
        #print('time spent on union graph = %f with adj matrix size %d' % (((un_end - un_start) / 60), len(adjmatrix)))
        #print('elapsed time = %f' % sumtime)

        del graph
        del adjmatrix
        del prunedadjm
        del maincore
    
    #stopword_weight_mat = calculateStopwordWeight(file_sum_mat, collection_terms)
    #stopwordsStats(stopword_weight_mat, collection_terms)
    ######################################
    print('****Union Graph stats********')
    print(nx.info(union_graph))
    # ---------------------------------- graph to weights -----------------------
    print("calculating Term weights")
    print('=======================')

    end = time.time()
    print('+++++++++++++++++++++++++++')
    print('TIME = %f MINS' % ((end - start) / 60))
    print('++++++++++++++++++++++++++++')

    # print(union_graph_termlist_id)
    # print(collection_terms)
    #graphToPng(union_graph)
    #graphToPng(gr)
    # print(Umatrix)
    # Wout : we will calculate it using the sum of weights on the graph edges on the fly
    # as Win we will use the lemma of GSB
    # -makris using the number of appearence of each terms on
    # collection scale and not the adj matrix of the union graph as its too big to use

    writetime = time.time()
    wout = Woutusinggraph(union_graph)
    print(postinglist)
    w_and_write_to_filev2(wout, collection_terms, union_graph_termlist_id, collection_term_freq, postinglist,
                          file=invfilename)
    endwritetime = time.time()
    print('**********************\n*creating  inverted index  in %f MINS \n***********************' % (
                (endwritetime - writetime) / 60))
    print('docs completely pruned %s'%str(docs_without_main_core))
#====>
elif menu == 2:
    # read files
    try:
        ids, trms, W, plist = load_inv_index('invertedindex.dat')  # index of terms GSB
        # - makris W
    except FileNotFoundError:
        invin = input('input inveted index for simple set based:')
        ids, trms, W, plist = load_inv_index(invin)
    try:
        docinfo = load_doc_info()
    except FileNotFoundError:
        docin = input('import documents index:')
        docinfo = load_doc_info(docin)
    try:
        invin = 'NegMain.dat'
        ids1, trms1, W1, plist1 = load_inv_index('NegMain.dat')  # index of terms W using maincore
    except FileNotFoundError:
        invin = input('input inveted index for NegMain implementation:')
        ids1, trms1, W1, plist1 = load_inv_index(invin)
    try:
        ids2, trms2, W2, plist2 = load_inv_index('PerSplit.dat')  # index of terms GSB
        # - makris W
    except FileNotFoundError:
        invin = input('input inveted index for PerSplit implementation:')
        ids2, trms2, W2, plist2 = load_inv_index(invin)
    try:
        ids3, trms3, W3, plist3 = load_inv_index('DotSplit.dat')  # index of terms GSB
        # - makris W
    except FileNotFoundError:
        invin = input('input inveted index for DotSplit implementation:')
        ids3, trms3, W3, plist3 = load_inv_index(invin)
    
    ######TEST###########
    try:
        ids4, trms4, W4, plist4 = load_inv_index('ConstantWindFile.dat')  # index of terms with constant window - makris W
    except FileNotFoundError:
        invin = input('input inveted index for constant window implementation:')
        ids4, trms4, W4, plist4 = load_inv_index(invin)
		
    try:
        ids5, trms5, W5, plist5 = load_inv_index('SenParConWind.dat')  # index of terms with constant sentence and paragraph window - makris W
    except FileNotFoundError:
        invin = input('input inveted index for constant sentence and paragraph window implementation:')
        ids5, trms5, W5, plist5 = load_inv_index(invin)
    #####################
    
    # debug
    #l = [i for i, j in zip(W, W1) if i == j]
    #print(l)