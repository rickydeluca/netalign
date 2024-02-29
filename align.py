from netalign.models.magna.magna import MAGNA
from netalign.models.pale.pale import PALE
from netalign.models.sigma.sigma import SIGMA
from netalign.models.stablegm.stablegm import StableGM
from netalign.models.ar.ar import AR
from netalign.models.shelley.shelley import SHELLEY


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
    elif 'ar' in cfg.MODEL.NAME.lower():
        model = AR(cfg)
    elif 'shelley' in cfg.MODEL.NAME.lower():
        model = SHELLEY(cfg)
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model.align(pair_dict)