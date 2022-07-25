import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as F2
import torchaudio.transforms as T
import numpy as np
import glob
from argparse import ArgumentParser
import os
import json
def estimateTimeDelay_files(audio_file_1, audio_file_2, 
                            offset_audio_file_1 = 0, 
                            lenght_audio_file_1 = -1, 
                            offset_audio_file_2 = 0, 
                            lenght_audio_file_2 = -1, 
                            downsample = 1):
    r"""
    Estimate the time delay of audio_file_2 with respect to audio_file_1.
    Input:
    - audio_file_1 (string): Audio file of reference
    - audio_file_2 (string): Audio file to estimate offset
    - offset_audio_file_1 (float): Offset audio_file_1 in seconds
    - lenght_audio_file_1 (float): Length audio_file_1 in seconds
    - offset_audio_file_2 (float): Offset audio_file_2 in seconds
    - lenght_audio_file_2 (float): Length audio_file_2 file in seconds
    Output:
    - estimated delay in second (float)
    - length of audio signal 1 in second (float)
    - length of audio signal 2 in second (float)
    Note:
    - Both signal should have the same sample rate
    - The signals can have diferent lenghts
    """

    # Read Metadata audio signals
    sample_rate_audio_file_1 = torchaudio.info(audio_file_1).sample_rate
    sample_rate_audio_file_2 = torchaudio.info(audio_file_2).sample_rate

    # Load audio signals
    signal_1, sample_rate_1 = torchaudio.load(audio_file_1, 
                                            frame_offset=offset_audio_file_1*sample_rate_audio_file_1, 
                                            num_frames= -1 if lenght_audio_file_1 == -1 else lenght_audio_file_1*sample_rate_audio_file_1)
    signal_2, sample_rate_2 = torchaudio.load(audio_file_2, 
                                            frame_offset=offset_audio_file_2*sample_rate_audio_file_2, 
                                            num_frames= -1 if lenght_audio_file_2 == -1 else lenght_audio_file_2*sample_rate_audio_file_2)
    signal_1 = signal_1[:,::downsample]
    signal_2 = signal_2[:,::downsample]
    sample_rate_1 = int(sample_rate_1/downsample)
    sample_rate_2 = int(sample_rate_2/downsample)

    # estiamnte delay between signals
    # delay = estimateTimeDelay_signals_time(signal_1[:,1000:50000], signal_2[:,1000:5000], start_offset=1000)
    # print(f'DELAY: {delay}')
    delay = estimateTimeDelay_signals(signal_1, signal_2)
    return delay/sample_rate_1, signal_1.shape[1]/sample_rate_1, signal_2.shape[1]/sample_rate_2


def estimateTimeDelay_signals(signal_1, signal_2):

    r"""
    Estimate the time delay of signal_1 with respect to signal_2 
    Input:
    - signal_1 (string): Signal of reference
    - signal_2 (string): Signal to estimate offset
    Output:
    - estimated delay in time-step (int)
    Note:
    - Both signal should have the same sample rate
    - The signals can have diferent lenghts
    """

    # the lenght of the signals after convolution
    conv_signal_length = signal_1.shape[1] + signal_2.shape[1] - 1

    # the last signal_ndim axes (1,2 or 3) will be transformed
    fft_1 = torch.fft.rfft(signal_1, conv_signal_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, conv_signal_length, dim=-1)

    # take the complex conjugate of one of the spectrums. 
    fft_multiplied = torch.conj(fft_1) * fft_2

    # back to time domain. 
    prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    final_result = torch.roll(prelim_correlation, (signal_1.shape[1],))

    # estimate the delay based on the length of reference signal
    delay = signal_1.shape[1] - torch.argmax(final_result)

    return delay

def verifyConsistency(ORIGINAL_WAV,
                      AUDIOVAULT_WAV,
                      segment_size = 100,
                      segment_nb = 10,
                      downsample = 100):

  
  info = torchaudio.info(ORIGINAL_WAV)
  duration = info.num_frames/info.sample_rate
  
  # if delay is None:
  offsets = []
  for i in range(segment_nb):
    offset_artificial = int((duration - segment_size)*i/segment_nb)
    delay, len1, len2 = estimateTimeDelay_files(audio_file_1 = ORIGINAL_WAV, 
                                                audio_file_2 = AUDIOVAULT_WAV,
                                                offset_audio_file_1 = 0,
                                                lenght_audio_file_1 = -1,
                                                offset_audio_file_2 = offset_artificial,
                                                lenght_audio_file_2 = segment_size,
                                                downsample = downsample)
    
    offsets.append((delay - offset_artificial).item())
    print(f" - [{offset_artificial:04d};{offset_artificial+segment_size:04d}]: delay={delay - offset_artificial:.3f}")
  return offsets


if __name__=='__main__':

    parser = ArgumentParser(description='Evaluate synchronization')
    # required arguments.
    parser.add_argument('--files_path', 
                        default='/ibex/scratch/projects/c2134/audiovault_data/processed', 
                        type=str, help='Path to all the folders')
    parser.add_argument('--segment_sizes', default=[1000,2000], type=int,
                        help='Segment sizes to run the alignment')
    parser.add_argument('--segment_nb', 
                        default=10, 
                        type=int, help='Number of segment to run it on')
    parser.add_argument('--downsample rates', 
                        default=[1, 10], 
                        type=int, help='ID of the list to run the alignment')
    parser.add_argument('--ID', required=True, type=int,
                        help='ID of the list to run the alignment')
    parser.add_argument('--out_path', 
                        default='/ibex/scratch/projects/c2134/audiovault_data/alignment_results',
                        type=str, help='Outpath to save files')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    all_files = sorted(glob.glob(f"{args.files_path}/*"))
    all_ids = list(map(lambda x: os.path.basename(x),all_files))
    ID_Movie = all_ids[args.ID]
    print(ID_Movie)
    ORIGINAL_WAV = f"{args.files_path}/{ID_Movie}/{ID_Movie}_original.wav"
    AUDIOVAULT_WAV = f"{args.files_path}/{ID_Movie}/{ID_Movie}.wav"
    
    results = {'parameters': {'segments':args.segment_sizes, 
                              'downsamples':args.downsamples,
                              'num_segments':args.segment_nb},
                f'{ID_Movie}': []}
    
    for segment in args.segment_sizes:
        for downsample in args.downsamples:
            offsets = verifyConsistency(ORIGINAL_WAV,
                                    AUDIOVAULT_WAV,
                                    segment_size = segment,
                                    segment_nb = args.segment_nb,
                                    downsample=downsample)
            results[f'{ID_Movie}'].append(offsets)

    # if this_mean:
    with open(f'{args.out_path}/{ID_Movie}_delay.json','w') as f:
        json.dump(results,f)