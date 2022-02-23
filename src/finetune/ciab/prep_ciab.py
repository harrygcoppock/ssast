# -*- coding: utf-8 -*-
# @Author  : Harry Coppock
# @Email   : harry.coppock@imperial.ac.uk
# @File    : prep_ciab.py

import numpy as np
import json
import os
import zipfile
import wget
from sklearn.model_selection import KFold

import pandas as pd
import subprocess, glob, csv
from tqdm import tqdm
import pickle
from botocore import UNSIGNED
from botocore.config import Config
import io
import boto3
import librosa, librosa.display
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import soundfile as sf

class PrepCIAB():
    POSSIBLE_MODALITIES = ['audio_sentence_url',
                           'audio_ha_sound_url',
                           'audio_cough_url',
                           'audio_three_cough_url']

    PATHS = {
            'pre_june_meta': 'combined-results/submissions/study_data_v6_pre_jun_11012022.pkl',
            'post_june_meta': 'combined-results/submissions/study_data_v6_post_jun_11012022.pkl',
            'meta_bucket': 'ciab-879281191186-prod-s3-pii-ciab-wip',
            'audio_bucket': 'ciab-879281191186-prod-s3-pii-ciab-approved',
            'splits': 'train-test-split/train_test_split_final_audiosentence_v6_final.pkl',
            'matched': 'audio_sentences_for_matching/test_set_matched_audio_sentences_v6_final.csv',
            'matched_train': 'audio_sentences_for_matching/train_set_matched_audio_sentences_v6_final.csv'
            }
    RANDOM_SEED = 42

    def __init__(self, modality='audio_three_cough_url'):
        self.modality = self.check_modality(modality)
        self.bucket_meta = self.get_bucket(self.PATHS['meta_bucket'])
        self.bucket_audio = self.get_bucket(self.PATHS['audio_bucket'])
        self.meta_data, self.train, self.test, self.long_test, self.matched_test, self.matched_train = self.load_train_test_splits()
        # base directory for audio files
        self.output_base= f'./data/ciab/{self.modality}'
        self.create_folds()

    def main(self):
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
            
        print('creating json')
        self.create_json()
        print('Beginining ciab train prepocessing')
        self.iterate_through_files(self.train, 'train')
        print('Beginining ciab test prepocessing')
        self.iterate_through_files(self.test, 'test')
        print('Beginining ciab long test prepocessing')
        self.iterate_through_files(self.long_test, 'long_test') 
        print('Beginining ciab matched test prepocessing')
        self.iterate_through_files(self.matched_test, 'matched_test') 
        print('Beginining ciab matched_train prepocessing')
        self.iterate_through_files(self.matched_train, 'matched_train')

    def check_modality(self, modality):
        if modality not in self.POSSIBLE_MODALITIES:
            raise Exception(f"{modaliity} is not one of the recorded functionalities,\
                                 please choose from {self.POSSIBLE_MODALITIES}")
        else:
            return modality


    def get_bucket(self, bucket_name):
        s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2')
        return s3_resource.Bucket(bucket_name)


    def get_file(self, path, bucket):
        return io.BytesIO(bucket.Object(path).get()['Body'].read())

    def load_train_test_splits(self):
        '''
        Loads the train and test barcode splits and the corresponding meta_data
        '''

        train_test = pd.read_pickle(self.get_file(
                                        self.PATHS['splits'],
                                         self.bucket_meta))
        pre_june_meta_data = pd.read_pickle(self.get_file(
                                        self.PATHS['pre_june_meta'],
                                        self.bucket_meta))
        post_june_meta_data = pd.read_pickle(self.get_file(
                                        self.PATHS['post_june_meta'],
                                        self.bucket_meta))
        meta_data = pd.concat([
                            pre_june_meta_data,
                            post_june_meta_data
                            ])
        long_test = self.create_long_test(meta_data, train_test)
        matched_test = pd.read_csv(self.get_file(
                                    self.PATHS['matched'],
                                    self.bucket_meta),
                                    names=['id'])
        matched_train = pd.read_csv(self.get_file(
                                    self.PATHS['matched_train'],
                                    self.bucket_meta),
                                    names=['id'])

        return meta_data, train_test['train'], train_test['test'], long_test.tolist(), matched_test['id'].tolist(), matched_train['id'].tolist()


    def iterate_through_files(self, dataset, split='train'):
        if not os.path.exists(f'{self.output_base}/audio_16k/{split}'):
            os.makedirs(f'{self.output_base}/audio_16k/{split}')
        self.error_list = []
        self.tot_removed = 0
        bootstrap_results = Parallel(n_jobs=-1, verbose=10, prefer='threads')(delayed(self.process_file)(barcode_id, split) for barcode_id in dataset)
        
        print(f'Average fraction removed: {mean(self.tot_removed)}')

    def process_file(self, barcode_id, split):

        df_match = self.meta_data[self.meta_data['audio_sentence'] == barcode_id]
        assert len(df_match) != 0, 'This unique code does not exist in the meta data file currently loaded - investigate!'
        try:
            filename = self.get_file(df_match[self.modality].iloc[0], self.bucket_audio)
        except:
            print(f"{df_match[self.modality].iloc[0]} not possible to load. From {df_match['processed_date']} Total so far: {len(error_list)}")
            self.error_list.append(df_match[self.modality].iloc[0])
            return 1
        label = df_match['test_result'].iloc[0]
        try:
            signal, sr = librosa.load(filename, sr=16000)
            #print('sox ' + filename + ' -r 16000 ' + self.output_base + '/audio_16k/'+ f"/{split}/" + barcode_id)
            #os.system('sox ' + filename + ' -r 16000 ' + self.output_base + '/audio_16k/'+ f"/{split}/" + barcode_id)
        except RuntimeError:
            print(f"{filename} not possible to load. From {df_match['processed_date']} Total so far: {len(error_list)}")
            self.error_list.append(filename)
            return 1
        clipped_signal, frac_removed = self.remove_silence(signal, barcode_id)
        self.tot_removed += frac_removed
        sf.write(f'{self.output_base}/audio_16k/{split}/{barcode_id}', clipped_signal, 16000)
        return 1
        
        #with open(f'{self.output_base}/audio_16k/{split}/errorlist.txt', "w") as output:
        #    output.write(str(error_list))


    def create_long_test(self, meta, train_test):
        '''
        Currently we are using any data collected after Novemenber 29th as a third test set.
        The collected barcodes are have not be listed however, can be determined by what barcodes are NOT in the train or test splits
        '''
        long_test = meta[meta['submission_time'] > '2021-11-29']
        long_test = long_test[long_test['test_result'] != 'Unknown/Void']
        long_test = long_test['audio_sentence']
        return long_test

    def print_stats(self):
        print(f'Sample numbers: Train: {len(self.train)}, Test: {len(self.test)}, Long_test: {len(self.long_test)} matched_test: {len(self.matched_test)}')
    
    def create_folds(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        self.folds = [[self.train[idx] for idx in test]
                 for (train, test) in kfold.split(self.train)]

    def create_json(self):
        for fold in [1,2,3,4,5]:
            train_list = [instance for instance in self.train if instance in self.folds[fold-1]]
            validation_list = [instance for instance in self.train if instance not in self.folds[fold-1]]
            
            with open('./data/datafiles/ciab_train_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': train_list}, f, indent=1)

            with open('./data/datafiles/ciab_validation_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': validation_list}, f, indent=1)
            with open('./data/datafiles/ciab_standard_test_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': self.test}, f, indent=1)
            with open('./data/datafiles/ciab_matched_test_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': self.matched_test}, f, indent=1)
            with open('./data/datafiles/ciab_long_test_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': self.long_test}, f, indent=1)

    def remove_silence(self, signal, filename):
        '''
        Removes the silent proportions of the signal, concatenating the remaining clips
        '''
        length_prior = len(signal)
        clips = librosa.effects.split(signal, top_db=60)

        clipped_signal = []
        for clip in clips:
            data = signal[clip[0]:clip[1]]
            clipped_signal.extend(data)
        length_post = len(clipped_signal)
        
        random_number = np.random.uniform(0,1,1)
        if random_number[0] < 0.1:

            self.plot_b_a(signal, np.array(clipped_signal), filename)

        return np.array(clipped_signal), (length_prior - length_post)/length_prior

    def plot_b_a(self, before, after, filename):
        '''
        plot the waveform before and after the silence is removed
        '''
        fig, ax = plt.subplots(nrows=2)
        librosa.display.waveshow(before, sr=16000, ax=ax[0])
        librosa.display.waveshow(after, sr=16000, ax=ax[1])
        ax[0].set(title='HOw much we remove')
        plt.savefig(f'figs/{filename}.png')
        plt.close()
if __name__ == '__main__':
    ciab = PrepCIAB()
    ciab.main()
#label_set = np.loadtxt('./data/esc_class_labels_indices.csv', delimiter=',', dtype='str')
#label_map = {}
#for i in range(1, len(label_set)):
#    label_map[eval(label_set[i][2])] = label_set[i][0]
#print(label_map)
#
## fix bug: generate an empty directory to save json files
#if os.path.exists('./data/datafiles') == False:
#    os.mkdir('./data/datafiles')
#
#for fold in [1,2,3,4,5]:
#    base_path = os.path.abspath(os.getcwd()) + "/data/ESC-50-master/audio_16k/"
#    meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
#    train_wav_list = []
#    eval_wav_list = []
#    for i in range(0, len(meta)):
#        cur_label = label_map[meta[i][3]]
#        cur_path = meta[i][0]
#        cur_fold = int(meta[i][1])
#        # /m/07rwj is just a dummy prefix
#        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
#        if cur_fold == fold:
#            eval_wav_list.append(cur_dict)
#        else:
#            train_wav_list.append(cur_dict)
#
#    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))
#
#    with open('./data/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
#        json.dump({'data': train_wav_list}, f, indent=1)
#
#    with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
#        json.dump({'data': eval_wav_list}, f, indent=1)
#
#print('Finished ESC-50 Preparation')
