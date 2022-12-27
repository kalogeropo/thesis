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