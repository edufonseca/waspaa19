
import numpy as np
import os
import utils
from numpy.random import permutation
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence, to_categorical



def get_one_hot(_all_labels, _patch_ids, n_classes):
    """
    get ground truth label (as one hot encoded vectors)
    given the entire dataset ground truth and a list of patch ids for the batch
    :param _all_labels:
    :param _patch_ids:
    :param n_classes:
    :return:
    """

    _y_int = np.empty((len(_patch_ids), 1), dtype='int')
    for tt in np.arange(len(_patch_ids)):
        _y_int[tt] = int(_all_labels[_patch_ids[tt]])
    _y_cat = to_categorical(_y_int, num_classes=n_classes)
    # ndarray (batch_size, n_classes)
    return _y_cat


def label_smoothing(y_train_cat, eps_LSR_noisy=0, delta_eps_LSR=None, num_classes=0, LSRmode=None):
    """
    Label smoothing
    y_train_cat: Categorical labels in the training data
    num_classes: number of classes in the training set
    delta_eps_LSR: delta epsilon to add to / substract from epsilon, based on a prior. This defines the final epsilon
    to be applied to the active class label
    """

    # {'Acoustic_guitar': 0, 'Bass_guitar': 1, 'Clapping': 2, 'Coin_(dropping)': 3, 'Crash_cymbal': 4,
    # 'Dishes_and_pots_and_pans': 5, 'Engine': 6, 'Fart': 7, 'Fire': 8, 'Fireworks': 9, 'Glass': 10, 'Hi-hat': 11,
    # 'Piano': 12, 'Rain': 13, 'Slam': 14, 'Squeak': 15, 'Tearing': 16, 'Walk_or_footsteps': 17, 'Wind': 18, 'Writing': 19}

    # GROUPS2: split classes in two groups based on amount of noise, according to acoustic inspection reported in
    # http: // www.eduardofonseca.net / FSDnoisy18k /
    # Noisier classes: bass guitar, clapping, crash, Engine, Fire, Rain, Slam, Walk, Wind
    idx_noisy_classes = [1, 2, 4, 6, 8, 13, 14, 17, 18]

    if LSRmode == 'GROUPS2' and delta_eps_LSR:
        # GROUPS2: split classes in two groups based on (more/less) noise. 2 levels of noisiness
        # some classes have higher/lower epsilon for the ACTIVE class label
        for kk in range(y_train_cat.shape[0]):
            if np.nonzero(y_train_cat[kk])[0] in idx_noisy_classes:
                # very noisy, increase epsi
                new_epsi = eps_LSR_noisy + delta_eps_LSR
            else:
                # low noise, decrease epsi
                new_epsi = eps_LSR_noisy - delta_eps_LSR
            y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + new_epsi/num_classes
    else:
        # standard LSR: all classes have the same epsilon for the ACTIVE class label
        y_train_cat = y_train_cat*(1-eps_LSR_noisy) + eps_LSR_noisy/num_classes

    return y_train_cat


def mixup(mode='intra', index=0, all_patch_indexes=None, batch_size=64, all_labels=None, all_features=None, alpha=0,
          n_classes=20):
    """
    Apply mixup, in two variants:
    - intra batch
    - inter batch
    """

    if mode == 'intra':
        patch_ids = all_patch_indexes[index * batch_size:(index + 1) * batch_size]

        # fetch labels for the batch
        # ndarray (batch_size, n_classes)
        _y_cat1 = get_one_hot(all_labels, patch_ids, n_classes)

        # fetch features for the batch
        # (batch_size, time, freq)
        _features1 = all_features[patch_ids]

        # create randomized copies for both patches and labels (with correspondence)
        patch_ids_rand = permutation(patch_ids)
        _y_cat2 = get_one_hot(all_labels, patch_ids_rand, n_classes)
        _features2 = all_features[patch_ids_rand]

    elif mode == 'inter':

        patch_ids1 = all_patch_indexes[index * batch_size:(index + 1) * batch_size]
        patch_ids2 = all_patch_indexes[(index + 1) * batch_size:(index + 2) * batch_size]
        if index == int(np.floor(len(all_patch_indexes)/batch_size) - 1):
            # if final batch, do mixup with the very first (else, incomplete remaining batch)
            patch_ids2 = all_patch_indexes[0:batch_size]

        _y_cat1 = get_one_hot(all_labels, patch_ids1, n_classes)
        _features1 = all_features[patch_ids1]

        _y_cat2 = get_one_hot(all_labels, patch_ids2, n_classes)
        _features2 = all_features[patch_ids2]

    # apply mixup, can be optmized, this is more readable
    y_cat_out = np.zeros_like(_y_cat1)
    _features = np.zeros_like(_features1)

    lam = np.random.beta(alpha, alpha, batch_size)

    for ii in range(batch_size):
        _features[ii] = lam[ii] * _features1[ii] + (1 - lam[ii]) * _features2[ii]
        y_cat_out[ii] = lam[ii] * _y_cat1[ii] + (1 - lam[ii]) * _y_cat2[ii]

    # adjust format to input CNN
    # (batch_size, 1, time, freq) for channels_first
    features_out = _features[:, np.newaxis]

    return features_out, y_cat_out


def get_label_files(filelist=None, dire=None, suffix_in=None, suffix_out=None):
    """

    :param filelist:
    :param dire:
    :param suffix_in:
    :param suffix_out:
    :return:
    """

    nb_files_total = len(filelist)
    labels = np.zeros((nb_files_total, 1), dtype=np.float32)
    for f_id in range(nb_files_total):
        labels[f_id] = utils.load_tensor(in_path=os.path.join(dire, filelist[f_id].replace(suffix_in, suffix_out)))
    return labels


# NOTE:
# these data generators work for small-medium size datasets under no memory constraints, eg RAM 32GB or more.
# If used with smaller RAMs, a slightly different approach for feeding the net may be needed.

class DataGeneratorPatch(Sequence):
    """
    Reads data from disk and returns batches.
    """

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.list_fnames = file_list
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))
        self.n_classes = int(params_learn.get('n_classes'))
        self.val_mode = False

        # LSR
        self.LSR = params_learn.get('LSR')
        self.eps_LSR_noisy = params_learn.get('eps_LSR_noisy')
        self.delta_eps_LSR = params_learn.get('delta_eps_LSR')
        self.LSRmode = params_learn.get('LSRmode')

        # mixup
        self.mixup = params_learn.get('mixup')
        self.mixup_mode = params_learn.get('mixup_mode')
        self.mixup_alpha = params_learn.get('mixup_alpha')


        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        # - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)

            # standardize the data
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if train set, create scaler, fit, transform, and save the scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)
                # this scaler will be used later on to scale val and test data

            else:
                self.val_mode = True
                # if we are in val or test set, load the training scaler as a param and transform
                self.features2d = scaler.transform(self.features2d)

            # after scaling in 2D, go back to tensor
            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        # but all the patches are contiguously ordered. shuffle them before making batches
        self.on_epoch_end()

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        - shuffle them
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')
        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances (or T_F patches) per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        # how many batches can we fit in the set
        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features and labels
        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):
        """
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features
        :param f_id:
        :return:
        """

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

    def __len__(self):
        return self.nb_iterations

    def __getitem__(self, index):
        """
        takes an index (batch number) and returns one batch of self.batch_size
        :param index:
        :return:
        """
        # index is taken care of by the Sequencer inherited
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # fetch labels for the batch
        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])
        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        if self.LSR and self.val_mode is False:
            # add label-smoothing fix improve
            y_cat = label_smoothing(y_cat,
                                    eps_LSR_noisy=self.eps_LSR_noisy,
                                    delta_eps_LSR=self.delta_eps_LSR,
                                    num_classes=self.n_classes,
                                    LSRmode=self.LSRmode,
                                    )

        # fetch features for the batch and adjust format to input CNN
        # (batch_size, 1, time, freq) for channels_first
        features = self.features[indexes, np.newaxis]

        if self.mixup and self.val_mode is False:
            features, y_cat = mixup(mode=self.mixup_mode,
                                    index=index,
                                    all_patch_indexes=self.indexes,
                                    batch_size=self.batch_size,
                                    all_labels=self.labels,
                                    all_features=self.features,
                                    n_classes=self.n_classes,
                                    alpha=self.mixup_alpha,
                                    )

        return features, y_cat


    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.nb_inst_total)


class PatchGeneratorPerFile(object):
    """
    Reads whole T_F representations from disk,
    and stores T_F patches *for a given entire file* in a tensor
    typically for prediction on a test set

    """

    def __init__(self, feature_dir=None, file_list=None, params_extract=None,
                 suffix_in='_mel', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        if feature_dir is not None:
            self.get_patches_features(feature_dir, file_list)

            # standardize the data: assuming this is used for inference
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if we are in val or test subset, load the training scaler as a param and transform
            self.features2d = scaler.transform(self.features2d)

            # go back to 3D tensor
            self.features = self.features2d.reshape(self.nb_patch_total, self.patch_len, self.feature_size)

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows or instances generated out of a given file
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"

        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data')]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_patch_total = self.nb_inst_cum[-1]

        # init current file, to keep track of the file yielded for prediction
        self.current_f_idx = 0

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features
        self.features = np.zeros((self.nb_patch_total, self.patch_len, self.feature_size), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):
        """
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features
        :param f_id:
        :return:
        """

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]  # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]  # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

    def get_patches_file(self):
        """
        Returns all the patches for one single audio clip
        """

        self.current_f_idx += 1
        # ranges form 1 to self.nb_files (ignores 0)
        assert self.current_f_idx <= self.nb_files, 'All the test files have been dispatched'

        # fetch features in the batch and adjust format to input CNN
        # (nb_patches_per_file, 1, time, freq)
        features = self.features[self.nb_inst_cum[self.current_f_idx-1]: self.nb_inst_cum[self.current_f_idx], np.newaxis]
        return features
