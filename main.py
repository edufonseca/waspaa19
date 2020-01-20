
#########################################################################
# Copyright Eduardo Fonseca 2019, v1.0
# This software is distributed under the terms of the MIT License
#
# If you use this code or part of it, please cite the following paper:
# Eduardo Fonseca, Frederic Font, and Xavier Serra, "Model-agnostic Approaches
# to Handling Noisy Labels When Training Sound Event Classifiers", in Proc.
# IEEE WASPAA 2019, NYC, USA, 2019
#
#########################################################################



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml
import shutil
import logging

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

import utils
from utils import GetCurrentEpoch
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from data import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile
from architectures import get_model_baseline, get_model_DenSE
from eval import Evaluator
from losses import lq_loss_wrap
from loss_time import TimeLossModel
import matplotlib
# matplotlib.get_backend()
matplotlib.use("TkAgg")
# matplotlib.use('agg')
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start = time.time()
now = datetime.datetime.now()
print("Current date and time:")
print(str(now))

# =========================================================================================================

# ==================================================================== ARGUMENTS
parser = argparse.ArgumentParser(description='Code for paper Model-agnostic Approaches to Handling Noisy Labels '
                                             'When Training Sound Event Classifiers')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))


# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables

# Read parameters file from yaml passed by argument
params = yaml.load(open(args.params_yaml))
params_ctrl = params['ctrl']
params_extract = params['extract']
params_learn = params['learn']
params_loss = params['loss']
params_recog = params['recognizer']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')


# determine loss function for stage 1 (or entire training)
if params_loss.get('type') == 'CCE':
    logger.info('Loss function for stage 1 (or entire training): CCE.')
    params_loss['type'] = 'categorical_crossentropy'
elif params_loss.get('type') == 'lq_loss':
    logger.info('Loss function for stage 1 (or entire training): Lq.')
    params_loss['type'] = lq_loss_wrap(params_loss.get('q_loss'))


params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))
#

# ======================================================== PATHS FOR DATA, FEATURES and GROUND TRUTH
# where to look for the dataset
path_root_data = params_ctrl.get('dataset_path')

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               'featuredir_tr': 'audio_train_varup2/',
               'featuredir_te': 'audio_test_varup2/',
               'path_to_dataset': path_root_data,
               'audiodir_tr': 'FSDnoisy18k.audio_train/',
               'audiodir_te': 'FSDnoisy18k.audio_test/',
               'audio_shapedir_tr': 'audio_train_shapes/',
               'audio_shapedir_te': 'audio_test_shapes/',
               'gt_files': os.path.join(path_root_data, 'FSDnoisy18k.meta')}


params_path['featurepath_tr'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_tr'))
params_path['featurepath_te'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_te'))

params_path['audiopath_tr'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_tr'))
params_path['audiopath_te'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_te'))

params_path['audio_shapepath_tr'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_tr'))
params_path['audio_shapepath_te'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_te'))


# ======================================================== SPECIFIC PATHS TO SOME IMPORTANT FILES
# ground truth
params_files = {'gt_test': os.path.join(params_path.get('gt_files'), 'test.csv'),
                'gt_train': os.path.join(params_path.get('gt_files'), 'train.csv')}

# # ============================================= print all params to keep record in output file
print('\nparams_ctrl=')
pprint.pprint(params_ctrl, width=1, indent=4)
print('params_files=')
pprint.pprint(params_files, width=1, indent=4)
print('params_extract=')
pprint.pprint(params_extract, width=1, indent=4)
print('params_learn=')
pprint.pprint(params_learn, width=1, indent=4)
print('params_loss=')
pprint.pprint(params_loss, width=1, indent=4)
print('params_recog=')
pprint.pprint(params_recog, width=1, indent=4)
print('\n')


# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA

# aim: lists with all wav files for tr and te
train_csv = pd.read_csv(params_files.get('gt_train'))
test_csv = pd.read_csv(params_files.get('gt_test'))
filelist_audio_tr = train_csv.fname.values.tolist()
filelist_audio_te = test_csv.fname.values.tolist()

# get positions of manually_verified clips: separate between CLEAN and NOISY sets
filelist_audio_tr_flagveri = train_csv.manually_verified.values.tolist()
idx_flagveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 1]
idx_flagnonveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 0]

# create list of ids that come from the noisy set
noisy_ids = [int(filelist_audio_tr[i].split('.')[0]) for i in idx_flagnonveri]
params_learn['noisy_ids'] = noisy_ids

# create dict with ground truth mapping with labels:
# -key: path to wav
# -value: the ground truth label too
file_to_label = {params_path.get('audiopath_tr') + k: v for k, v in
                 zip(train_csv.fname.values, train_csv.label.values)}

# ========================================================== CREATE VARS FOR DATASET MANAGEMENT
# list with unique n_classes labels and aso_ids
list_labels = sorted(list(set(train_csv.label.values)))

# create dicts such that key: value is as follows
# label: int
# int: label
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

# create ground truth mapping with categorical values
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}


# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# compute T_F representation
# mel-spectrogram for all files in the dataset and store it

if params_ctrl.get('feat_ext'):
    n_extracted_tr = 0; n_extracted_te = 0; n_failed_tr = 0; n_failed_te = 0

    # only if features have not been extracted, ie
    # if folder does not exist, or it exists with less than 80% of the feature files
    # create folder and extract features
    nb_files_tr = len(filelist_audio_tr)
    if not os.path.exists(params_path.get('featurepath_tr')) or \
                    len(os.listdir(params_path.get('featurepath_tr'))) < nb_files_tr*0.8:

        if os.path.exists(params_path.get('featurepath_tr')):
            shutil.rmtree(params_path.get('featurepath_tr'))
        os.makedirs(params_path.get('featurepath_tr'))

        print('\nFeature extraction for {} train set..........................................'.format(
            params_ctrl.get('train_data')))

        for idx, f_name in enumerate(filelist_audio_tr):
            f_path = os.path.join(params_path.get('audiopath_tr'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path,
                                    input_fixed_length=params_extract['audio_len_samples'],
                                    params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_mel')

                # save also label
                utils.save_tensor(var=np.array([file_to_int[f_path]], dtype=float),
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_label')

                if os.path.isfile(os.path.join(params_path.get('featurepath_tr'),
                                               f_name.replace('.wav', suffix_in + '.data'))):
                    n_extracted_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted tr features', (idx + 1), nb_files_tr, f_path))
                else:
                    n_failed_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract tr features', (idx + 1), nb_files_tr, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this tr audio is in the csv but not in the folder', (idx + 1), nb_files_tr, f_path))

        print('n_extracted_tr: {0} / {1}'.format(n_extracted_tr, nb_files_tr))
        print('n_failed_tr: {0} / {1}\n'.format(n_failed_tr, nb_files_tr))

    else:
        print('Train set {} is already extracted in {}'.format(params_ctrl.get('train_data'),
                                                               params_path.get('featurepath_tr')))

    if not os.path.exists(params_path.get('featurepath_te')):
        os.makedirs(params_path.get('featurepath_te'))

        print('\nFeature extraction for test set ............................................')

        nb_files_te = len(filelist_audio_te)
        for idx, f_name in enumerate(filelist_audio_te):
            f_path = os.path.join(params_path.get('audiopath_te'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path,
                                    input_fixed_length=params_extract['audio_len_samples'],
                                    params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_te'),
                                                          f_name.replace('.wav', '.data')), suffix='_mel')

                if os.path.isfile(os.path.join(params_path.get('featurepath_te'),
                                               f_name.replace('.wav', '_mel.data'))):
                    n_extracted_te += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted te features', (idx + 1), nb_files_te, f_path))
                else:
                    n_failed_te += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract te features', (idx + 1), nb_files_te, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this te audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))

        print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
        print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))

    else:
        print('Test set is already extracted in {}'.format(params_path.get('featurepath_te')))


# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION

# Assuming features or T-F representations on a per-file fashion previously computed and in disk
# input: '_mel'
# output: '_label'

# select the noisy set of training data
if params_ctrl.get('train_data') == 'noisy':
    # only files (not path), feature file list for tr, only those that are NOT verified: NOISY SET
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_flagnonveri]
else:
    raise ValueError('This work uses the noisy set of FSDnoisy18k')

# get label for every file *from the .data saved in disk*, in float
labels_audio_train = get_label_files(filelist=ff_list_tr,
                                     dire=params_path.get('featurepath_tr'),
                                     suffix_in=suffix_in,
                                     suffix_out=suffix_out
                                     )

print('Number of clips considered as train set: {0}'.format(len(ff_list_tr)))
print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

# split the val set randomly (but stratified) within the train set
tr_files, val_files = train_test_split(ff_list_tr,
                                       test_size=params_learn.get('val_split'),
                                       stratify=labels_audio_train,
                                       random_state=42
                                       )

tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                  file_list=tr_files,
                                  params_learn=params_learn,
                                  params_extract=params_extract,
                                  suffix_in='_mel',
                                  suffix_out='_label',
                                  floatx=np.float32
                                  )

val_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                   file_list=val_files,
                                   params_learn=params_learn,
                                   params_extract=params_extract,
                                   suffix_in='_mel',
                                   suffix_out='_label',
                                   floatx=np.float32,
                                   scaler=tr_gen_patch.scaler
                                   )


# ============================================================DEFINE AND FIT A MODEL
# ============================================================DEFINE AND FIT A MODEL
# ============================================================DEFINE AND FIT A MODEL

tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
# ============================================================
if params_ctrl.get('learn'):
    if params_learn.get('model') == 'baseline':
        model = get_model_baseline(params_learn=params_learn, params_extract=params_extract)
    elif params_learn.get('model') == 'DenSE':
        model = get_model_DenSE(params_learn=params_learn, params_extract=params_extract)

if params_learn.get('mode') == 0:
    # implementing a warmup period for mixup*******************************************************
    logger.info('Using mode 0: warmup based mixup.')

    opt = Adam(lr=params_learn.get('lr'))
    model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
    model.summary()

    logger.info('===WARMUP STAGE (no mixup)**************************.')

    # delete previous generator to free memory
    del tr_gen_patch

    params_learn['mixup'] = False
    tr_gen_patch_mixup_warmup = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                                   file_list=tr_files,
                                                   params_learn=params_learn,
                                                   params_extract=params_extract,
                                                   suffix_in='_mel',
                                                   suffix_out='_label',
                                                   floatx=np.float32
                                                   )

    # callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [reduce_lr]

    hist1 = model.fit_generator(tr_gen_patch_mixup_warmup,
                                steps_per_epoch=tr_gen_patch_mixup_warmup.nb_iterations,
                                epochs=params_learn.get('mixup_warmup_epochs'),
                                validation_data=val_gen_patch,
                                validation_steps=val_gen_patch.nb_iterations,
                                class_weight=None,
                                workers=4,
                                verbose=2,
                                callbacks=callback_list)

    logger.info('===FINAL STAGE (with mixup)**************************')

    # delete previous generator to free memory
    del tr_gen_patch_mixup_warmup

    params_learn['mixup'] = True
    tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                      file_list=tr_files,
                                      params_learn=params_learn,
                                      params_extract=params_extract,
                                      suffix_in='_mel',
                                      suffix_out='_label',
                                      floatx=np.float32
                                      )

    # callbacks
    early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [early_stop, reduce_lr]

    hist2 = model.fit_generator(tr_gen_patch,
                                steps_per_epoch=tr_gen_patch.nb_iterations,
                                initial_epoch=params_learn.get('mixup_warmup_epochs'),
                                epochs=params_learn.get('n_epochs'),
                                validation_data=val_gen_patch,
                                validation_steps=val_gen_patch.nb_iterations,
                                class_weight=None,
                                workers=4,
                                verbose=2,
                                callbacks=callback_list)

    hist1.history['acc'].extend(hist2.history['acc'])
    hist1.history['val_acc'].extend(hist2.history['val_acc'])
    hist1.history['loss'].extend(hist2.history['loss'])
    hist1.history['val_loss'].extend(hist2.history['val_loss'])
    hist1.history['lr'].extend(hist2.history['lr'])
    hist = hist1

elif params_learn.get('mode') == 1:
    # implementing standard training*******************************************************
    logger.info('Using mode 1: standard training, including LSR, standard mixup, and non time-dependent loss function.')

    opt = Adam(lr=params_learn.get('lr'))
    model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
    model.summary()

    # callbacks
    early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [early_stop, reduce_lr]

    hist = model.fit_generator(tr_gen_patch,
                               steps_per_epoch=tr_gen_patch.nb_iterations,
                               epochs=params_learn.get('n_epochs'),
                               validation_data=val_gen_patch,
                               validation_steps=val_gen_patch.nb_iterations,
                               class_weight=None,
                               workers=4,
                               verbose=2,
                               callbacks=callback_list)


elif params_learn.get('mode') == 2:
    # implementing time-dependent noise robust loss function***********************************************
    logger.info('Using mode 2: time-dependent loss function.')

    # two-stage learning process **************************************************************************
    timelossmodel = TimeLossModel(model=model, params_learn=params_learn, params_loss=params_loss)
    timelossmodel.compile()

    # callbacks
    get_current_epoch = GetCurrentEpoch(current_epoch=timelossmodel.current_epoch)
    early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [get_current_epoch, reduce_lr, early_stop]

    hist = timelossmodel.model.fit_generator(tr_gen_patch,
                                             steps_per_epoch=tr_gen_patch.nb_iterations,
                                             epochs=params_learn.get('n_epochs'),
                                             validation_data=val_gen_patch,
                                             validation_steps=val_gen_patch.nb_iterations,
                                             class_weight=None,
                                             workers=4,
                                             verbose=2,
                                             callbacks=callback_list)

elif params_learn.get('mode') == 3:
    logger.info('Using mode 3: prune train set with current model checkpoint after learn.prune_stage1 epochs.')

    opt = Adam(lr=params_learn.get('lr'))
    model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
    model.summary()

    logger.info('STAGE 1: Train with Lq for n1 epoch on entire train set========================================')

    # callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [reduce_lr]

    hist1 = model.fit_generator(tr_gen_patch,
                                steps_per_epoch=tr_gen_patch.nb_iterations,
                                epochs=params_learn.get('prune_stage1'),
                                validation_data=val_gen_patch,
                                validation_steps=val_gen_patch.nb_iterations,
                                workers=4,
                                verbose=2,
                                class_weight=None,
                                callbacks=callback_list)

    logger.info('Predict on the TRAIN set: predictions at patch level:========================================')

    tr_scaler = tr_gen_patch.scaler
    del tr_gen_patch
    tr_gen_patch_for_prediction = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_tr'),
                                                        file_list=tr_files,
                                                        params_extract=params_extract,
                                                        suffix_in='_mel',
                                                        floatx=np.float32,
                                                        scaler=tr_scaler
                                                        )

    from_clip_to_loss_clip = {item: [] for item in tr_files}
    losses_train_set_clip = [None] * len(tr_files)

    for i in trange(len(tr_files), miniters=int(len(tr_files) / 100), ascii=True, desc="Predicting..."):
        # return all patches for a sound file
        patches_file = tr_gen_patch_for_prediction.get_patches_file()
        preds_patch_list = model.predict(patches_file).tolist()
        preds_patch = np.array(preds_patch_list)

        # get gt as one-hot-vector to compute losses
        label = utils.load_tensor(in_path=os.path.join(params_path.get('featurepath_tr'),
                                                       tr_files[i].replace('_mel', '_label')))
        y_true = to_categorical(int(label[0]), num_classes=params_learn.get('n_classes'))

        # from predictions at patch level to loss values at patch level (one real value per patch)
        if params_learn.get('prune_loss_type') == 'lq_loss':
            y_pred = np.clip(preds_patch, params_extract.get('eps'), 1. - params_extract.get('eps'))
            q = params_learn.get('prune_loss_q_value')
            tmp = y_pred * y_true
            loss = np.max(tmp, axis=-1)
            prune_loss = (1 - (loss + 10 ** (-8)) ** q) / q

        # aggregate patch level loss values to one loss value per clip in train set
        if params_learn.get('prune_aggregate_loss_clip') == 'amean':
            from_clip_to_loss_clip[tr_files[i]] = np.mean(prune_loss, axis=0)
            losses_train_set_clip[i] = np.mean(prune_loss, axis=0)

    losses_train_set = np.array(losses_train_set_clip)
    logger.info('Use loss values to prune dataset ========================================')

    # define upper threshold for pruning
    if params_learn.get('prune_loss_threshold_method') == 'percentile':
        thres = np.percentile(losses_train_set, params_learn.get('prune_discard_percentile'))
        logger.info('Pruning through percentile {}'.format(params_learn.get('prune_discard_percentile')))

    # losses higher than thres are discarded through pruning
    tr_files_after_pruning = []
    for k, v in from_clip_to_loss_clip.items():
        if v <= thres:
            tr_files_after_pruning.append(k)

    logger.info('Number of clips kept for stage 2 after pruning: {}'.format(len(tr_files_after_pruning)))
    logger.info('Number of clips pruned: {}'.format(len(tr_files) - len(tr_files_after_pruning)))

    logger.info('STAGE 2: Train with Lq until convergence on tr_files_after_pruning===================================')

    del tr_gen_patch_for_prediction
    tr_gen_patch_pruned = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                             file_list=tr_files_after_pruning,
                                             params_learn=params_learn,
                                             params_extract=params_extract,
                                             suffix_in='_mel',
                                             suffix_out='_label',
                                             floatx=np.float32
                                             )

    # callbacks
    early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [early_stop, reduce_lr]

    hist2 = model.fit_generator(tr_gen_patch_pruned,
                                steps_per_epoch=tr_gen_patch_pruned.nb_iterations,
                                initial_epoch=params_learn.get('prune_stage1'),
                                epochs=params_learn.get('n_epochs'),
                                validation_data=val_gen_patch,
                                validation_steps=val_gen_patch.nb_iterations,
                                class_weight=None,
                                workers=4,
                                verbose=2,
                                callbacks=callback_list)

# ==================================================================================================== PREDICT
# ==================================================================================================== PREDICT
# ==================================================================================================== PREDICT

logger.info('Compute predictions on test set:==================================================\n')

list_preds = []
te_files = [f for f in os.listdir(params_path.get('featurepath_te')) if f.endswith(suffix_in + '.data')]
te_preds = np.empty((len(te_files), params_learn.get('n_classes')))

# grab every T_F file (computed on the file level) - split it in T_F patches and store it in tensor, sorted by file
try:
    te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_te'),
                                         file_list=te_files,
                                         params_extract=params_extract,
                                         suffix_in='_mel',
                                         floatx=np.float32,
                                         scaler=tr_gen_patch.scaler
                                         )
except:
    te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_te'),
                                         file_list=te_files,
                                         params_extract=params_extract,
                                         suffix_in='_mel',
                                         floatx=np.float32,
                                         scaler=tr_scaler
                                         )

for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
    # return all patches for a sound file
    patches_file = te_gen_patch.get_patches_file()
    preds_patch_list = model.predict(patches_file).tolist()
    preds_patch = np.array(preds_patch_list)

    # aggregate softmax values across patches in order to produce predictions on the file/clip level
    if params_learn.get('predict_agg') == 'amean':
        preds_file = np.mean(preds_patch, axis=0)
    elif params_recog.get('aggregate') == 'gmean':
        preds_file = gmean(preds_patch, axis=0)
    else:
        raise ValueError('unkown aggregation method for prediction')

    te_preds[i, :] = preds_file

list_labels = np.array(list_labels)
pred_label_files_int = np.argmax(te_preds, axis=1)
pred_labels = [int_to_label[x] for x in pred_label_files_int]

# create dataframe with predictions
# columns: fname & label
# this is based on the features file, instead on the wav file
te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in os.listdir(params_path.get('featurepath_te'))
                if f.endswith(suffix_in + '.data')]
pred = pd.DataFrame(te_files_wav, columns=["fname"])
pred['label'] = pred_labels

# # =================================================================================================== EVAL
# # =================================================================================================== EVAL
# # =================================================================================================== EVAL
print('\nEvaluate ACC and print score============================================================================')

# read ground truth
gt_test = pd.read_csv(params_files.get('gt_test'))

# init Evaluator object
evaluator = Evaluator(gt_test, pred, list_labels, params_ctrl, params_files)

print('\n=============================ACCURACY===============================================================')
print('=============================ACCURACY===============================================================\n')
evaluator.evaluate_acc()
evaluator.evaluate_acc_classwise()
evaluator.print_summary_eval()

end = time.time()
print('\n=============================Job finalized==========================================================\n')
print('Time elapsed for the job: %7.2f hours' % ((end - start) / 3600.0))
print('\n====================================================================================================\n')
