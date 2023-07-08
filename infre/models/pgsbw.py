from infre.models import GSBWindow, BaseIRModel
from infre.helpers.functions import prune_graph

class PGSBW(GSBWindow, BaseIRModel):
    def __init__(self, collection, window, clusters):

        GSBWindow.__init__(self, collection, window)

        # model name
        self.model = self._model()

        # empty graph to be filled by union
        self.graph, _ = prune_graph(self.graph, collection, clusters)

        # NW Weight of GSBs
        # _nwk() is calcualted twice due to inheritance. Better way to be found!
        self._nwk()


    def _model(self): return __class__.__name__