from .base import BaseDataset, Data

class ComposeDatasets(BaseDataset):
    def __init__(self, datasets):
        self.data_list = []
        
        for i in datasets:
            if self.data_list != []:
                if self.data_lists.h.shape[1] != i.data_lists.h.shape[1] or self.data_lists.g.shape[1] != i.data_lists.g.shape[1]:
                    print("error")
            self.data_list += i.data_list
        
    def process(self, **input_params):
        raise NotImplementedError