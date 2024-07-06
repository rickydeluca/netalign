from netalign.models.bigalign import BigAlign
from netalign.models.isorank import IsoRank
from netalign.models.deeplink import DeepLink
from netalign.models.final import FINAL
from netalign.models.ione import IONE
from netalign.models.magna import MAGNA
from netalign.models.pale import PALE
from netalign.models.shelley import SHELLEY, SHELLEY_G

__all__ = ['BigAlign', 'DeepLink', 'IsoRank', 'FINAL', 'MAGNA', 'PALE', 'SHELLEY', 'SHELLEY_G']

def init_align_model(cfg):
    """
    Init the alignment model wrt the configuration
    dictionary (`cfg`).
    """

    if cfg.NAME.lower() == 'isorank':
        model = IsoRank(cfg)
        name = f'isorank'
    elif cfg.NAME.lower() == 'bigalign':
        model = BigAlign(cfg)
        name = f'bigalign_l{cfg.LAMBDA}'
    elif cfg.NAME.lower() == 'deeplink':
        model = DeepLink(cfg)
        name = f'deeplink'
    elif cfg.NAME.lower() == 'final':
        model = FINAL(cfg)
        name = f'final'
    elif cfg.NAME.lower() == 'ione':
        model = IONE(cfg)
        name = f'ione'
    elif cfg.NAME.lower() == 'magna':
        model = MAGNA(cfg)
        name = f'magna_{cfg.MEASURE}-p{cfg.POPULATION_SIZE}-g{cfg.NUM_GENERATIONS}'
    elif cfg.NAME.lower() == 'pale':
        model = PALE(cfg)
        name = f'pale_h{cfg.MAPPING.NUM_HIDDEN}-{cfg.MAPPING.LOSS_FUNCTION}'
    elif cfg.NAME.lower() == 'shelley':
        model = SHELLEY(cfg)
        name = f'shelley_{cfg.FEATS.TYPE}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}'
    elif cfg.NAME.lower() == 'shelleyg':
        model = SHELLEY_G(cfg)
        name = f'shelleyg_{cfg.FEATS.TYPE}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}'
    else:
        raise ValueError(f'Invalid model: {cfg.NAME.lower()}')
    
    return model, name