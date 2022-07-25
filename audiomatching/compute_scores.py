import json
import statistics
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

path = '/ibex/scratch/projects/c2134/audiovault_data'

json_file = json.load(open(f"{path}/all_delays.json"))

# flat_results = {'results':{k:[delay for delays in v for delay in delays] for k,v in json_file.items() if k!='parameters'}}
flat_results = {'results':{k:v[2] for k,v in json_file.items() if k!='parameters'}}

# flat_results['parameters'] = json_file['parameters']

def compute_score(delays, time_th=0.1):
    return np.sum(np.abs(np.array(delays) - statistics.median(delays)) <= time_th)/len(delays)


for th in np.linspace(0,1.0,11):
    good_movies = {k:statistics.median(delays) for k,delays in flat_results['results'].items() if compute_score(delays) >=th}
    # diffs = [abs(statistics.median(v) - statistics.mode(v)) for k,v in flat_results['results'].items()]
    print(f"{th*100:05.1f}%: {len(good_movies)}")


th = 1
good_movies = {k:statistics.median(delays) for k,delays in flat_results['results'].items() if compute_score(delays) >=th}
# diffs = [abs(statistics.median(v) - statistics.mode(v)) for k,v in flat_results['results'].items()]
# print(f"{th}: {len(good_movies)}")
with open(f'{path}/audiovault_aligned.json', 'w') as f:
    json.dump(good_movies,f,indent=4)

plot = False
if plot:
    max_diff = 0.2
    diffs = [x if x<max_diff else max_diff for x in diffs]
    # the histogram of the data
    n, bins, patches = plt.hist(diffs, 50, density=False, facecolor='g', alpha=0.75)


    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(0, 100)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.savefig('hist_diff.jpg')