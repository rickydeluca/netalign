from netalign.mapping.magna.magna import MAGNA
from netalign.mapping.pale.pale import PALE
from netalign.mapping.sigma.sigma import SIGMA
from netalign.mapping.stablegm.stablegm import StableGM


def align_networks(pair_dict, cfg):

    # Get alignment model
    if 'magna' in cfg.MODEL.NAME.lower():
        model = MAGNA(cfg)
    elif 'pale' in cfg.MODEL.NAME.lower():
        model = PALE(cfg)
    elif 'sigma' in cfg.MODEL.NAME.lower():
        model = SIGMA(cfg)
    elif 'stablegm' in cfg.MODEL.NAME.lower():
        model = StableGM(cfg)
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model.align(pair_dict)