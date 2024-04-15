from netalign.models import MAGNA, PALE, SIGMA, SHELLEY_G, SHELLEY_N, IsoRank


def align_networks(pair_dict, cfg):

    # Get alignment model.
    model_name = cfg.MODEL.NAME.lower()

    if model_name == 'isorank':
        model = IsoRank(
            alpha=cfg.MODEL.ALPHA,
            maxiter=cfg.MODEL.MAX_ITER,
            tol=cfg.MODEL.TOL
        )
    elif model_name == 'magna':
        model = MAGNA(
            measure=cfg.MODEL.MEASURE,
            population_size=cfg.MODEL.POPULATION_SIZE,
            num_generations=cfg.MODEL.NUM_GENERATIONS
        )
    elif 'pale' in cfg.MODEL.NAME.lower():
        model = PALE(cfg)
    elif 'sigma' in cfg.MODEL.NAME.lower():
        model = SIGMA(cfg)
    elif 'shelley' in cfg.MODEL.NAME.lower():
        model = SHELLEY_N(cfg)
    else:
        raise ValueError(f'Invalid alignment model: {cfg.MODEL.NAME}.')
    
    # Align networks
    return model(pair_dict)