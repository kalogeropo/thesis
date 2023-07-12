from infre.models import SetBased, GSB, GSBWindow
from numpy import mean
from infre.preprocess import Collection
from infre.xlwriter import ExcelWriter
import os
import yaml
import pandas as pd


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
    # experiment = pd.DataFrame()

    # ############ SET BASED ####################
    # sb_model = SetBased(col).fit(queries)
    # pre, rec = sb_model.evaluate(rels)
    # print(f'SetBased: {mean(pre):.3f}, {mean(rec):.3f}')
    
    # # store experiment
    # experiment['SB'] = pre
    
    # ############ GRAPHICAL SET BASED #################### 
    # gsb_model = GSB(col).fit(queries)
    # pre, rec = gsb_model.evaluate(rels)
    # print(f'GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    
    # # store experiment
    # experiment['GSB'] = pre

    ########## GRAPHICAL SET BASED Constant WIWNDOW ####################
    w = config["window"]["size"]
    
    gsb_window_model = GSBWindow(col, window=w).fit(queries)
    pre, rec = gsb_window_model.evaluate(rels)
    print(f'GSBW: {mean(pre):.3f}, {mean(rec):.3f}')

    # store experiment
    # experiment[f'GSBW-{w}'] = pre

    # print("Storing Experiments...")
    # writer.write_to_excel(sheet_name='old-models', dataframe=experiment)
    # print(experiment)