from infre.models import GSB, PGSB, ConGSB
from numpy import mean
from infre.preprocess import Collection
import pandas as pd
from infre.xlwriter import ExcelWriter
# configs & other
import yaml
import os


if __name__ == '__main__':

    # load configuration
    config_path = os.path.join(os.getcwd(), "config.yml")

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load anad create collection
    col_path = config["path"]["collection"]
    first_n_docs = config["first_n"]
    col = Collection(col_path).create(first=first_n_docs)

    # load associated queries and relevant documents
    qr_path = config["path"]["qr_path"]
    queries, rels = Collection.load_qd(qr_path)

    # apriori min frequncy for valid termsets
    min_freq = config["min_frequency"]

    # save experiment handler
    file_path = config["file_path"]
    writer = ExcelWriter(file_path)


    # dataframe to store each experiment (precision of the model) as a nenw column
    experiment = pd.DataFrame()

    # keep track of pruned edges for future and further analysis
    avg_precs = []
    prune_perc = []

    # values to be tested
    cluster_sizes = config["cluster_sizes"]
    similaritiess = config["similarities"]

    total_exps = len(cluster_sizes) * len(similaritiess) * 2 # (2 is the number of models)
    current_exp = 0
    for cluster_size in cluster_sizes:
        for similarity in similaritiess:
            
            current_exp += 1
            print(f"Calculating {current_exp / total_exps} experiments...")
            
            ################ GRAPHICAL SET BASED WITH PRUNING (PGSB) ####################  
            pgsb_model = PGSB(col, clusters=cluster_size, condition={'sim': similarity}).fit(queries)
            pre, rec = pgsb_model.evaluate(rels)
            print(f'PGSB: {mean(pre):.3f}, {mean(rec):.3f}')

            # get average precision and percentage of pruning applied
            avg_precs += [mean(pre)]
            prune_perc += [pgsb_model.prune]

            # concatenate new experiments
            experiment[f'PGSB-{cluster_size}-{similarity}'] = pre


            ############### CONCEPTUALIZED GRAPHICAL SET BASED ####################
            con_gsb_model = ConGSB(col, clusters=cluster_size, cond={'sim': similarity}).fit(queries)
            pre, rec = con_gsb_model.evaluate(rels)
            print(f'CGSB: {mean(pre):.3f}, {mean(rec):.3f}')   

            # get average precision and percentage of pruning applied
            avg_precs += [mean(pre)]
            prune_perc += [con_gsb_model.prune]

            # concatenate new experiment
            experiment[f'CGSB-{cluster_size}-{similarity}'] = pre

    experiment['avg_pre'] = pd.Series(avg_precs)
    experiment['%prune'] = pd.Series(prune_perc)

    print("Storing Experiments...")
    writer.write_to_excel(sheet_name='complete-graph-clusters-sim', dataframe=experiment)