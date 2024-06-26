from netalign.models.isorank import IsoRank
from netalign.models.magna import MAGNA
from netalign.models.pale import PALE
from netalign.models.shelley import SHELLEY
from netalign.models.sigma import SIGMA

__all__ = ['IsoRank', 'MAGNA', 'PALE', 'SHELLEY', 'SIGMA']

def init_align_model(cfg):
    """
    Init the alignment model wrt the configuration
    dictionary (`cfg`).
    """
    
    if cfg.NAME.lower() == 'magna':
        model = MAGNA(cfg)
        name = f'magna_{cfg.MEASURE}-p{cfg.POPULATION_SIZE}-g{cfg.NUM_GENERATIONS}'
    elif cfg.NAME.lower() == 'pale':
        model = PALE(cfg)
        name = f'pale_h{cfg.MAPPING.NUM_HIDDEN}-{cfg.MAPPING.LOSS_FUNCTION}'
    elif cfg.NAME.lower() == 'shelley':
        model = SHELLEY(cfg)
        name = f'shelley_{cfg.FEATS.TYPE}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}'
    else:
        raise ValueError(f'Invalid model: {cfg.MODEL.NAME.lower()}')
    
    return model, name