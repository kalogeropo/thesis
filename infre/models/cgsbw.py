from infre.models import GSBWindow, ConGSB
from infre.helpers.functions import prune_graph

# IMPORTANT: due to __mro__, ConGSBWindow searches ConGSB methods first
# a.k.a, ConGSBWindow leverages methods from ConGSB and not GSBWindow
class ConGSBWindow(ConGSB, GSBWindow):
    def __init__(self, collection, window, clusters, cond={}):
        GSBWindow.__init__(self, collection, window)
        # ConGSB.__init__(self, collection, cond)

        self.model = self._model()
        
        self.graph, self.embeddings, self.prune = prune_graph(self.graph, collection, n_clstrs=clusters, condition=cond)
        
        self._nwk()
        
        self._cnwk()

    def _model(self): return __class__.__name__