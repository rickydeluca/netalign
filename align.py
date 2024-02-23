from netalign.mapping.magna.magna import MAGNA
from netalign.mapping.pale.pale import PALE
from netalign.mapping.tau.tau import Tau

def align_networks(pair_dict, cfg, device='cpu'):

    # Get alignment model
    if 'magna' in cfg.MODEL.NAME.lower():
        model = MAGNA(cfg)
    elif 'pale' in cfg.MODEL.NAME.lower():
        model = PALE(cfg).to(device)
    elif 'tau' in cfg.MODEL.NAME.lower():
        model = Tau(cfg).to(device)   
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model.align(pair_dict)