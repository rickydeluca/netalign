from netalign.models.magna.magna import MAGNA
from netalign.models.pale.pale import PALE

def align_networks(pair_dict, cfg):
    # Get alignment model
    if 'magna' in cfg.MODEL.NAME.lower():
        model = MAGNA(cfg)
    elif 'pale' in cfg.MODEL.NAME.lower():
        model = PALE(cfg)
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model.align(pair_dict)