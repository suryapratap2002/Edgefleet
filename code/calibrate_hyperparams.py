import json

cfg = {
    'tracker': {'process_var': 1e-2, 'meas_var': 20.0},
    'hsv': {'lower': [5, 90, 80], 'upper': [30, 255, 255]},
    'detector': {'conf_thresh': 0.25, 'tile_conf_thresh': 0.18}
}
with open('calibrations.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Wrote calibrations.json. Edit it to tune parameters.')
