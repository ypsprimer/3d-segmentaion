def check_multi_val(config):
    split = config['prepare']['val_split']
    if isinstance(split, list) and len(split) > 1:
        return True, len(split)
    else:
        return False, 1
