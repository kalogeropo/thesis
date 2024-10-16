import os
import pandas as pd

# configs & other
import yaml
from numpy import array, mean

from infre.preprocess import Collection
from infre.xlwriter import ExcelWriter


def run_experiment(models, c={}):

    # load configuration
    config_path = os.path.join(os.getcwd(), "config.yml")

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load and create collection
    col_path = config["path"]["collection"]
    first_n_docs = config["first_n"]
    col = Collection(col_path).create(first=first_n_docs)

    # Load associated queries and relevant documents
    qr_path = config["path"]["qr_path"]
    queries, rels = Collection.load_qd(qr_path)

    # Save experiment handler
    file_path = config["file_path"]
    writer = ExcelWriter(file_path)

    # Dataframe to store each experiment (precision of the model) as a new column
    experiment = pd.DataFrame()

    # Keep track of pruned edges for future and further analysis
    avg_precs = []
    prune_perc = []

    try:
        cond = list(c.keys())[0]
        typ = list(c.values())[0]

        if cond == "edge":
            wout_avg = config["wout_avg"][typ]
            second_param = array(config["edge_coeffs"]) * wout_avg
        elif cond == "sim":
            second_param = config["similarities"]

    except IndexError:
        second_param = []

    cluster_sizes = config["cluster_sizes"]

    total_exps = len(cluster_sizes) * len(second_param) * len(models)
    current_exp = 0
    for cluster_size in cluster_sizes:
        for j, param in enumerate(second_param):
            current_exp += 1
            print(f"Calculating {current_exp / total_exps} experiments...")

            for i in range(len(models)):
                # Create model instance using the provided constructor
                model = models[i](
                    col, clusters=cluster_size, condition={cond: param}
                ).fit(queries)
                pre, rec = models[i].evaluate(rels)
                print(f"{model._model_name()}: {mean(pre):.3f}, {mean(rec):.3f}")

                # Get average precision and percentage of pruning applied
                avg_precs += [mean(pre)]
                prune_perc += [model.prune]

                # Concatenate new experiments
                experiment[f"{model._model_name()}-{cluster_size}-{param}"] = pre

    experiment["avg_pre"] = pd.Series(avg_precs)
    experiment["%prune"] = pd.Series(prune_perc)

    print("Storing Experiments...")
    writer.write_to_excel(
        sheet_name="complete-graph-clusters-sim", dataframe=experiment
    )
