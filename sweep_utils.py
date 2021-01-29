import math

def set_updates_by_batch_sz(config, num_samples):
    if '--update-freq' in config:
        uf = config['--update-freq'].current_value
    else:
        uf = 1

    updates = num_samples / config['--batch-size'].current_value * config['--max-epoch'].current_value / uf
    updates = int(math.ceil(updates))
    warmup_updates = int(updates * 0.1)
    config['--max-update'].current_value = updates
    config['--lr-period-updates'].current_value = updates - warmup_updates
    config['--warmup-updates'].current_value = warmup_updates