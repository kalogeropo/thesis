from json import JSONEncoder
from pandas import ExcelWriter
from numpy import integer, floating, ndarray, int32



# on creating a graph index, to cast int32 to int for JSON graph indexing
class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, integer):
            return int(obj)
        if isinstance(obj, floating):
            return float(obj)
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, int32):
            return int(obj)
        return JSONEncoder.default(self, obj)


# Here are the queries and relevant parser. It is important to notice that, this class does NOT handle the collection
# parsing, but the *.txt files which are created by the neccessary collection parsing scripts



#TODO: CREATE file with test name as name if not exists, Fix warning on writer.save()
class excelwriter():
    def __init__(self, path):
        self.res_path = path

    def write_results(self, sheet_name, df):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        try:
            writer = ExcelWriter("".join([self.res_path,sheet_name,'.xlsx']), engine='openpyxl', mode="a", if_sheet_exists='new')
        except FileNotFoundError:
            writer = ExcelWriter("".join([self.res_path,sheet_name,'.xlsx']), engine='openpyxl')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
