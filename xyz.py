import numpy as np, pprint
results = np.load('./MLAC_results/detection_results_with_metrics.npy', allow_pickle=True).item()
for layer, metrics in results.items():
    print(f'\n{layer}')
    pprint.pprint(metrics)