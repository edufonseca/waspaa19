
from keras import backend as K
import tensorflow as tf


def lq_loss_wrap(_q):
    def lq_loss_core(y_true, y_pred):
        """
        This loss function is proposed in:
         Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
         Noisy Labels", 2018
        https://arxiv.org/pdf/1805.07836.pdf
        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_q)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q

        return _loss
    return lq_loss_core