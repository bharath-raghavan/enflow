import torch

ELEMENTS = { 1:'H' ,                                                                                                                                                                             2:'He',
             3:'Li',   4:'Be',                                                                                                                 5:'B' ,   6:'C' ,   7:'N' ,   8:'O' ,   9:'F' ,  10:'Ne',
            11:'Na',  12:'Mg',                                                                                                                13:'Al',  14:'Si',  15:'P' ,  16:'S' ,  17:'Cl',  18:'Ar',
            19:'K' ,  20:'Ca',  21:'Sc',            22:'Ti',  23:'V' ,  24:'Cr',  25:'Mn',  26:'Fe',  27:'Co',  28:'Ni',  29:'Cu',  30:'Zn',  31:'Ga',  32:'Ge',  33:'As',  34:'Se',  35:'Br',  36:'Kr',
            37:'Rb',  38:'Sr',  39:'Y' ,            40:'Zr',  41:'Nb',  42:'Mo',  43:'Tc',  44:'Ru',  45:'Rh',  46:'Pd',  47:'Ag',  48:'Cd',  49:'In',  50:'Sn',  51:'Sb',  52:'Te',  53:'I' ,  54:'Xe',
            55:'Cs',  56:'Ba',  57:'La',            72:'Hf',  73:'Ta',  74:'W' ,  75:'Re',  76:'Os',  77:'Ir',  78:'Pt',  79:'Au',  80:'Hg',  81:'Tl',  82:'Pb',  83:'Bi',  84:'Po',  85:'At',  86:'Rn',
            87:'Fr',  88:'Ra',  89:'Ac',           104:'Rf', 105:'Db', 106:'Sg', 107:'Bh', 108:'Hs', 109:'Mt', 110:'Ds', 111:'Rg', 112:'Cn', 113:'Nh', 114:'Fl', 115:'Mc', 116:'Lv', 117:'Ts', 118:'Og',
                                          58:'Ce',  59:'Pr',  60:'Nd',  61:'Pm',  62:'Sm',  63:'Eu',  64:'Gd',  65:'Tb',  66:'Dy',  67:'Ho',  68:'Er',  69:'Tm',  70:'Yb',  71:'Lu',
                                          90:'Th',  91:'Pa',  92:'U' ,  93:'Np',  94:'Pu',  95:'Am',  96:'Cm',  97:'Bk',  98:'Cf',  99:'Es', 100:'Fm', 101:'Md', 102:'No', 103:'Lr'
}

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

def get_element(elem, mass):
    if elem == '':
        mass_int = int(round(mass))
        if mass_int == 1:  # Guess H from mass
            return 'H'
        elif mass_int < 36 and mass_int > 1:  # Guess He to Cl from mass
            return ELEMENTS[mass_int//2]
        else:
            print("error")
    
    return elem
        
def one_hot(index, num_classes=None, dtype=None):
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)
