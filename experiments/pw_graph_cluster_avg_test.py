from infre.models import PGSBW, ConGSBWindow
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

    avg_precs = []
    prune_perc = []

    cluster_sizes = config["cluster_sizes"][-4:]
    percentages = config["percentages"]
    coeffs = config["edge_coeffs"][-2:]
    avg_wouts = config["wout_avg"]["pw"]

    total_exps = len(cluster_sizes) * len(percentages) * len(coeffs) * 2 # (2 is the number of models)
    current_exp = 0
    for perc, avg_wout in zip(percentages, avg_wouts):
        for cluster_size in cluster_sizes:
            for coeff in coeffs:

                current_exp += 2
                print(f"Calculating {current_exp} / {total_exps}...")
                print(f'Experiment: (Window-{perc} - {cluster_size} - {coeff})')
            
                ################ GRAPHICAL SET BASED Window Perc WITH PRUNING (PGSBW) ####################  
                pgsbw_model = PGSBW(col, window=perc, clusters=cluster_size, condition={"edge":avg_wout*coeff}).fit(queries)
                pre, rec = pgsbw_model.evaluate(rels)
                print(f'PGSBW: {mean(pre):.3f}, {mean(rec):.3f}')

                # get average precision and percentage of pruning applied
                avg_precs += [mean(pre)]
                prune_perc += [pgsbw_model.prune]

                # concatenate new experiment
                experiment[f'PGSBW-{perc}-{cluster_size}-{coeff}'] = pre


                ############### CONCEPTUALIZED GRAPHICAL SET BASED Perc Window ####################
                con_gsbw_model = ConGSBWindow(col, window=perc, clusters=cluster_size, cond={"edge":avg_wout*coeff}).fit(queries)
                pre, rec = con_gsbw_model.evaluate(rels)
                print(f'CGSBW: {mean(pre):.3f}, {mean(rec):.3f}')   

                # get average precision and percentage of pruning applied
                avg_precs += [mean(pre)]
                prune_perc += [con_gsbw_model.prune]

                # concatenate new experiment
                experiment[f'CGSBW-{perc}-{cluster_size}-{coeff}'] = pre


    experiment['avg_pre'] = pd.Series(avg_precs)
    experiment['%prune'] = pd.Series(prune_perc)

    print("Storing Experiments...")
    writer.write_to_excel(sheet_name='pw-graph-clusters-avg', dataframe=experiment)