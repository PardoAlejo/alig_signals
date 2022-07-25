# AUDIO ALIGNMENT MAD DATASET

The project contains two main scripts:



**align_signals.py**: Align signals from two files using different downsampling rates and store the results of such aligment on a json file. 

```
Input: A pair of audio files.
```

**compute_scores.py:** Computes scores for each of the alignments from the previous json file and gives a score to divide alignment as good or bad alignments. The script expects a single json file with all the aligments.

```
Input: A single json file with all the aligments computed by align_signals.py.
```

### Observations

The missaligned audios can be better aligned by changing the `sample rates` and `segment sizes` in `align_signals.py`. **It's a semi-automatic process, and requires constant human verification.**
