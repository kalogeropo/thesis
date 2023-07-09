from infre.models import GSB, BaseIRModel
from infre.helpers.functions import prune_graph

class PGSB(GSB, BaseIRModel):
    def __init__(self, collection, clusters, condition={}):

        BaseIRModel.__init__(self, collection)
        
        # model name
        self.model = self._model()

        self.graph, _, self.prune = prune_graph(self.union_graph(), collection, clusters, condition=condition)

        # NW Weight of GSBs
        self._nwk()

    def _model(self): return __class__.__name__