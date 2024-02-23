from netalign.mapping.magna.magna import MAGNA
from netalign.mapping.pale.pale import PALE

def align_networks(pair_dict, cfg, device='cpu'):

    # Get alignment model
    if 'magna' in cfg.MODEL.NAME.lower():
        model = MAGNA(cfg)
    elif 'pale' in cfg.MODEL.NAME.lower():
        model = PALE(cfg)
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model.align(pair_dict)