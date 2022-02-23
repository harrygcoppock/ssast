
# -*- coding: utf-8 -*-
# @Author  : Harry Coppock
# @Email   : harry.coppock@imperial.ac.uk
# @File    : calculate_mean_std.py
from ....dataloader import AudioDataset


CalcMeanStd(AudioDataset):
'''
need to transform to fbank than calculate the mean and std from this - use the audio dataset functionality for this. the audio conf should matach what is in run_ciab.sh

'''
    def __init__():
        pass



audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}
