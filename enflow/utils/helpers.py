import torch
    
def get_box(dataset):
    def _get_box(pos):
        # get bounding box
        min_ = torch.min(pos, dim=0)[0].abs()
        max_ = torch.max(pos, dim=0)[0].abs()
        return torch.max(torch.stack((min_,max_)), dim=0)[0].round()*2 
    
    box = torch.tensor([0, 0, 0])

    for data in dataset.data_list:
        current_box = _get_box(data.pos)
        box = torch.max(torch.stack((box, current_box)), dim=0)[0]
    
    return box
    
def one_hot(index, num_classes=None, dtype=None):
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)
