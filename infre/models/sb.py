from infre.models import BaseIRModel

class SetBased(BaseIRModel):
    def __init__(self, collection=None):
        
        # inherit from base model
        super().__init__(collection)

        # model name
        self.model = self._model()


    def _model(self): return __class__.__name__


    def _model_func(self, termsets): return 
    
    
    # tsf * idf for set based
    def _vectorizer(self, tsf_ij, idf, *args):
        ########## each column corresponds to a document #########
        return tsf_ij * idf.reshape(-1, 1)