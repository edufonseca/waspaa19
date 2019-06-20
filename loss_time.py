
# keras
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
import numpy as np


class TimeLossModel(object):

    def __init__(self, model=None, params_learn=None, params_loss=None):
        # Hyper parameters
        self.params_learn = params_learn
        self.params_loss = params_loss

        # Model is built outside and passed as argument
        self.model = model

        # Optimizer
        self.optimizer = Adam(lr=params_learn.get('lr'))

        # Loss Function params*********************
        # n1 epochs in stage 1 of the learning process
        self.n1 = params_learn.get('stage1_epoch')

        # this is for the combination of 2 functions in time*************************************
        if params_loss.get('type') == 'lq_lqmax_time_fade':
            self.q1 = params_loss.get('q_loss')
            self.q2 = params_loss.get('q_loss2')
            self.r2 = params_loss.get('m_loss2')
            self.transition = params_loss.get('transition')
            self.loss_func = self.lq_lqmax_time_fade_wrap()


    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['accuracy'])
        print('Model Compiled!')
        self.model.summary()

    # we combine over time 2 loss functions that do not change over time

    # STAGE 1*********************************************************************************************************
    # STAGE 1*********************************************************************************************************

    def lq_loss(self, y_true, y_pred):
        print("\n==lq_loss")

        # hyper param
        print('q1:', self.q1)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** self.q1) / self.q1
        return _loss

    # STAGE 2*********************************************************************************************************
    # STAGE 2*********************************************************************************************************

    # lq loss but thresholded **************************************************************************
    def lq_loss_max(self, y_true, y_pred):
        print("\n==lq_loss_max")

        # hyper param
        print('q2:', self.q2)
        print('r:', self.r2)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss = (1 - (_loss + 10 ** (-8)) ** self.q2) / self.q2

        # threshold
        t_m = K.max(_loss) * self.r2
        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32'))
        _loss = _loss * _mask_m

        return _loss

    # CLOSURE or WRAPPER*********************************************************************************************
    # CLOSURE or WRAPPER*********************************************************************************************

    def lq_lqmax_time_fade_wrap(self):
        # watch these are functions, for stage 1 and 2
        loss_1 = self.lq_loss
        loss_2 = self.lq_loss_max
        print('\n===lq_lqmax_time_fade_wrap')

        # init the tensor with current epoch, to be updated during training, and define var in scope
        self.current_epoch = K.variable(0.0)
        current_epoch = self.current_epoch

        # define alpha: ratio between both losses
        # these values are pre-computed; other transitions could be tried.
        if self.transition == 'fade25_30':
            slope = -0.2
            cte = 6
        elif self.transition == 'fade25_35':
            slope = -0.1
            cte = 3.5

        # alpha is in [inf:-inf], but it is bounded within the loss function
        # its crosses with 1 and 0 are given by a precomputed straight line, ie the transition
        alpha = current_epoch * slope + cte
        print('self.transition:', self.transition)
        print('slope:', slope)
        print('cte:', cte)
        print('alpha:', alpha)

        def lq_lqmax_time_fade_core(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            print('\n===lq_lqmax_time_fade_wrap.lq_lqmax_time_fade_core')

            early_loss = loss_1(y_true, y_pred)
            late_loss = loss_2(y_true, y_pred)

            # fade
            mask_1 = K.min([K.max([alpha, 0]), 1])
            mask_2 = 1 - mask_1
            combined_loss = mask_1 * early_loss + mask_2 * late_loss

            # scale the losses to normalize only by the number of non-zero loss values in the batch
            # compute number of non-zero loss values in the batch, after the discard
            n_non_zero = K.sum(K.cast(K.greater(combined_loss, 0), K.floatx()))
            # multiply by batch_size (64), to neutralize the corresponding division of the mean operation outside this function
            # divide by n_non_zero
            # essentially we do: sum(all the losses) / number of non-zero losses
            combined_loss_norma = combined_loss * 64.0 / n_non_zero
            return combined_loss_norma
        return lq_lqmax_time_fade_core

