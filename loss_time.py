from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


class TimeLossModel(object):

    def __init__(self, model=None, params_learn=None, params_loss=None):
        # Hyper parameters
        self.params_learn = params_learn
        self.params_loss = params_loss

        # Model is built outside and passed as argument
        self.model = model

        self.optimizer = Adam(lr=params_learn.get('lr'))

        # Loss Function params*********************
        # n1 epochs in stage 1 of the learning process
        self.n1 = params_learn.get('stage1_epoch')

        # combination of 2 functions in time*************************************
        if params_loss.get('type') == 'lq_lqmax_time_sudden':
            self.q1 = params_loss.get('q_loss')
            self.q2 = params_loss.get('q_loss2')
            self.r2 = params_loss.get('m_loss2')
            self.loss_func = self.lq_lqmax_time_sudden_wrap()

        if params_loss.get('type') == 'lq_lqperc_time_sudden':
            self.q1 = params_loss.get('q_loss')
            self.q2 = params_loss.get('q_loss2')
            self.perc2 = params_loss.get('perc_loss2')
            self.loss_func = self.lq_lqperc_time_sudden_wrap()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['accuracy'])
        print('Model Compiled!')
        self.model.summary()

    # STAGE 1*********************************************************************************************************
    # STAGE 1*********************************************************************************************************

    def lq_loss(self, y_true, y_pred):
        print("\n==lq_loss")

        # hyper param
        print('q1:', self.q1)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** self.q1) / self.q1
        return _loss

    # STAGE 2*********************************************************************************************************
    # STAGE 2*********************************************************************************************************

    def lq_loss_max(self, y_true, y_pred):
        """
        lq loss but thresholded with m.max
        :param y_true:
        :param y_pred:
        :return:
        """
        print("\n==lq_loss_max")

        # hyper param
        print('q2:', self.q2)
        print('r:', self.r2)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** self.q2) / self.q2

        # threshold
        t_m = K.max(_loss) * self.r2
        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32'))
        _loss = _loss * _mask_m

        return _loss

    def lq_loss_perc(self, y_true, y_pred):
        """
        lq loss but thresholded with percentile

        :param y_true:
        :param y_pred:
        :return:
        """
        print("\n==lq_loss_perc")

        # hyper param
        print('q2:', self.q2)
        print('perc2:', self.perc2)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** self.q2) / self.q2

        # threshold
        # compute the percentile X of the distribution: previously defined perc2 makes sure to discard 1,2,3,4,5 patches
        # interpolation lower to be consistent with pre-design (else, default is nearest, which keeps changing)
        t_m = tf.contrib.distributions.percentile(_loss, q=self.perc2, interpolation='lower')
        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32'))
        _loss = _loss * _mask_m

        return _loss

    # CLOSURE or WRAPPER*********************************************************************************************
    # CLOSURE or WRAPPER*********************************************************************************************

    def lq_lqmax_time_sudden_wrap(self):
        loss_1 = self.lq_loss
        loss_2 = self.lq_loss_max
        print('\n===lq_lqmax_time_sudden_wrap')

        # init the tensor with current epoch, to be updated during training, and define var in scope
        self.current_epoch = K.variable(0.0)
        current_epoch = self.current_epoch

        # define hparams in scope: stage 1 length in the loss function.
        _n1 = self.n1

        def lq_lqmax_time_sudden_core(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            print('\n===lq_lqmax_time_sudden_wrap.lq_lqmax_time_sudden_core')
            print('n1:', _n1)

            early_loss = loss_1(y_true, y_pred)
            late_loss = loss_2(y_true, y_pred)

            # choose the current loss as a function of epoch, changing suddenly at n1 epochs
            mask_1 = K.cast(K.less(current_epoch, _n1), 'float32')
            mask_2 = K.cast(K.greater_equal(current_epoch, _n1), 'float32')
            combined_loss = mask_1 * early_loss + mask_2 * late_loss

            # scale the losses to normalize only by the number of non-zero loss values in the batch
            # compute number of non-zero loss values in the batch, after the discard
            n_non_zero = K.sum(K.cast(K.greater(combined_loss, 0), K.floatx()))
            # multiply by 64 = batch_size, to neutralize division of K.mean operation outside this function
            # divide by n_non_zero
            # essentially: sum(all the losses) / number of non-zero losses
            combined_loss_norma = combined_loss * 64.0 / n_non_zero
            return combined_loss_norma
        return lq_lqmax_time_sudden_core

    def lq_lqperc_time_sudden_wrap(self):
        loss_1 = self.lq_loss
        loss_2 = self.lq_loss_perc
        print('\n===lq_lqperc_time_sudden_wrap')

        # init the tensor with current epoch, to be updated during training, and define var in scope
        self.current_epoch = K.variable(0.0)
        current_epoch = self.current_epoch

        # define hparams in scope: stage 1 length in the loss function.
        _n1 = self.n1

        def lq_lqperc_time_sudden_core(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            print('\n===lq_lqperc_time_sudden_wrap.lq_lqperc_time_sudden_core')
            print('n1:', _n1)

            early_loss = loss_1(y_true, y_pred)
            late_loss = loss_2(y_true, y_pred)

            # choose the current loss as a function of epoch, changing suddenly at n1 epochs
            mask_1 = K.cast(K.less(current_epoch, _n1), 'float32')
            mask_2 = K.cast(K.greater_equal(current_epoch, _n1), 'float32')
            combined_loss = mask_1 * early_loss + mask_2 * late_loss

            # scale the losses to normalize only by the number of non-zero loss values in the batch
            # compute number of non-zero loss values in the batch, after the discard
            n_non_zero = K.sum(K.cast(K.greater(combined_loss, 0), K.floatx()))
            # multiply by 64 = batch_size, to neutralize division of K.mean operation outside this function
            # divide by n_non_zero
            # essentially: sum(all the losses) / number of non-zero losses
            combined_loss_norma = combined_loss * 64.0 / n_non_zero
            return combined_loss_norma
        return lq_lqperc_time_sudden_core

