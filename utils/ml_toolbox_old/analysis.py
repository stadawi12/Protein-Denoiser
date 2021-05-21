from datetime import datetime
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend, Model
from tensorflow.keras.layers import Input, Conv3D, Activation, MaxPooling3D, Conv3DTranspose, UpSampling3D, concatenate
from tensorflow.python.ops import math_ops, clip_ops
from tensorflow.keras.utils import to_categorical

from viewer import plot_metric, confusion_matrix


def class_occurence(data, mode='penal_weight', labs=None, per_sample=False, save=False, file_appendix='',
                    verbose=False):
    modes = ['nums', 'props', 'plain_weight', 'penal_weight']

    if str(type(data)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
        ys = data.numpy()
    elif isinstance(data, list) and str(type(data[0])) == "<class 'proteins.Sample'>":
        #print('Data is tiled?: ',data[0].has_tiles)
        if not data[0].has_tiles:
            ys = np.asarray([d.lab for d in data])
        else:
            # if tiles present need to count in tiles as maps are not the same size (will fail in unique)
            # this will overestimate the counts due to overlaps between tiles
            ys = np.asarray([ltile for d in data for ltile in d.ltiles])
    else:
        raise RuntimeError("Parameter 'data' should be either a np.array, a list or tf.Tensor type.")

    if labs is None:
        vals, count = np.unique(ys, return_counts=True)
    else:
        vals = np.asarray(labs)
        count = np.array([np.size(ys[ys == u]) for u in vals])

    if per_sample and str(type(data)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
        raise RuntimeError("Parameter 'per_sample' is only supported if data is an instance of 'Sample' class.")
    if per_sample:
        if verbose:
            print("\n-------------------------- CLASS PROPORTIONS PER CLASS {}:".format(file_appendix))
        ids = [d.id for d in data]
        # _, count = list(zip(*[np.unique(y, return_counts=True) for y in ys]))
        # print(count)
        c_count = np.array([[np.size(y[y == u]) for u in vals] for y in ys])
        c_props = (c_count / np.reshape(np.sum(c_count, axis=1), (len(c_count), 1))) * 100
        props_df = pd.DataFrame(c_props, index=ids, columns=vals.astype(int)).round(3)
        if verbose:
            print(props_df)
        if save:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            props_df.to_csv(r'logs/class_occurence_per_class{}.csv'.format(
                '_' + file_appendix if file_appendix is not '' else file_appendix))

    props = count / np.sum(count)
    #print(props)
    if mode == 'nums':
        if verbose:
            print("\n-------------------------- CLASS OCCURRENCES {}:".format(file_appendix))
        ret = dict(zip(vals.astype(int), count))
    elif mode == 'props':
        if verbose:
            print("\n-------------------------- CLASS PROPORTIONS {}:".format(file_appendix))
        ret = dict(zip(vals.astype(int), props * 100))
        #print(ret)
    elif mode == 'plain_weight':
        if verbose:
            print("\n-------------------------- CLASS WEIGHTS (SIMPLE) {}:".format(file_appendix))
        ret = (1 - props) / np.sum(1 - props)
        ret = dict(zip(vals.astype(int), ret))
    elif mode == 'penal_weight':
        if verbose:
            print("\n-------------------------- CLASS WEIGHTS (PENALISE) {}:".format(file_appendix))
        ret = np.nan_to_num((1 / count) / np.sum(1 / count[np.isfinite(1 / count)]))
        ret = dict(zip(vals.astype(int), ret))
    else:
        raise RuntimeError("Mode must be one of {}".format(modes))

    ret_df = pd.DataFrame.from_dict(ret, orient='index', columns=[mode]).round(3)
    if verbose:
        print(ret_df)
    if save:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        ret_df.to_csv(r'logs/class_occurence{}.csv'.format(
            '_' + file_appendix if file_appendix is not '' else file_appendix))

    return ret_df.values.T[0]


def unet_small(inpt, labs, up=False, feats=False):
    conv1_1 = Conv3D(32, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_1")(inpt)
    relu1_1 = Activation('relu', name='activation_1')(conv1_1)
    conv1_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_2")(relu1_1)
    relu1_2 = Activation('relu', name='activation_2')(conv1_2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_1")(relu1_2)

    conv2_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_3")(pool1)
    relu2_1 = Activation('relu', name="activation_3")(conv2_1)
    conv2_2 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_4")(relu2_1)
    relu2_2 = Activation('relu', name="activation4")(conv2_2)

    if not up:
        up3 = Conv3DTranspose(128, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_3")(relu2_2)
    else:
        up3 = UpSampling3D((2, 2, 2))(relu2_2)
    add3 = concatenate([up3, relu1_2], axis=1, name="concatenate_3")
    conv3_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_13")(add3)
    relu3_1 = Activation('relu', name="activation_13")(conv3_1)
    conv3_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_14")(relu3_1)
    relu3_2 = Activation('relu', name="activation_14")(conv3_2)

    if not feats:
        last_layer = Conv3D(labs, (1, 1, 1), name="conv3d_15")(relu3_2)
    else:
        last_layer = relu3_2
    return last_layer


def unet_medium(inpt, labs, up=False, feats=False):
    conv1_1 = Conv3D(32, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_1")(inpt)
    relu1_1 = Activation('relu', name='activation_1')(conv1_1)
    conv1_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_2")(relu1_1)
    relu1_2 = Activation('relu', name='activation_2')(conv1_2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_1")(relu1_2)

    conv2_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_3")(pool1)
    relu2_1 = Activation('relu', name="activation_3")(conv2_1)
    conv2_2 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_4")(relu2_1)
    relu2_2 = Activation('relu', name="activation4")(conv2_2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_2")(relu2_2)

    conv3_1 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_5")(pool2)
    relu3_1 = Activation('relu', name="activation_5")(conv3_1)
    conv3_2 = Conv3D(256, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_6")(relu3_1)
    relu3_2 = Activation('relu', name="activation_6")(conv3_2)

    if not up:
        up4 = Conv3DTranspose(256, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_1")(relu3_2)
    else:
        up4 = UpSampling3D((2, 2, 2))(relu3_2)
    add4 = concatenate([up4, relu2_2], axis=1, name="concatenate_1")
    conv4_1 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_7")(add4)
    relu4_1 = Activation('relu', name="activation_7")(conv4_1)
    conv4_2 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_8")(relu4_1)
    relu4_2 = Activation('relu', name="activation_8")(conv4_2)

    if not up:
        up5 = Conv3DTranspose(128, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_2")(relu4_2)
    else:
        up5 = UpSampling3D((2, 2, 2))(relu4_2)
    add5 = concatenate([up5, relu1_2], axis=1, name="concatenate_2")
    conv5_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_9")(add5)
    relu5_1 = Activation('relu', name="activation_9")(conv5_1)
    conv5_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_10")(relu5_1)
    relu5_2 = Activation('relu', name="activation_10")(conv5_2)

    if not feats:
        last_layer = Conv3D(labs, (1, 1, 1), name="conv3d_11")(relu5_2)
    else:
        last_layer = relu5_2
    return last_layer


def unet_large(inpt, labs, up=False, feats=False):
    conv1_1 = Conv3D(32, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_1")(inpt)
    relu1_1 = Activation('relu', name='activation_1')(conv1_1)
    conv1_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_2")(relu1_1)
    relu1_2 = Activation('relu', name='activation_2')(conv1_2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_1")(relu1_2)

    conv2_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_3")(pool1)
    relu2_1 = Activation('relu', name="activation_3")(conv2_1)
    conv2_2 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_4")(relu2_1)
    relu2_2 = Activation('relu', name="activation4")(conv2_2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_2")(relu2_2)

    conv3_1 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_5")(pool2)
    relu3_1 = Activation('relu', name="activation_5")(conv3_1)
    conv3_2 = Conv3D(256, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_6")(relu3_1)
    relu3_2 = Activation('relu', name="activation_6")(conv3_2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), name="max_pooling3d_3")(relu3_2)

    conv4_1 = Conv3D(256, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_7")(pool3)
    relu4_1 = Activation('relu', name="activation_7")(conv4_1)
    conv4_2 = Conv3D(512, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_8")(relu4_1)
    relu4_2 = Activation('relu', name="activation_8")(conv4_2)

    if not up:
        up5 = Conv3DTranspose(512, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_1")(relu4_2)
    else:
        up5 = UpSampling3D((2, 2, 2))(relu4_2)
    add5 = concatenate([up5, relu3_2], name="concatenate_1")
    conv5_1 = Conv3D(256, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_9")(add5)
    relu5_1 = Activation('relu', name="activation_9")(conv5_1)
    conv5_2 = Conv3D(256, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_10")(relu5_1)
    relu5_2 = Activation('relu', name="activation_10")(conv5_2)

    if not up:
        up6 = Conv3DTranspose(256, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_2")(relu5_2)
    else:
        up6 = UpSampling3D((2, 2, 2))(relu5_2)
    add6 = concatenate([up6, relu2_2], name="concatenate_2")
    conv6_1 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_11")(add6)
    relu6_1 = Activation('relu', name="activation_11")(conv6_1)
    conv6_2 = Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_12")(relu6_1)
    relu6_2 = Activation('relu', name="activation_12")(conv6_2)

    if not up:
        up7 = Conv3DTranspose(128, (2, 2, 2), padding='same', strides=(2, 2, 2), name="conv3d_transpose_3")(relu6_2)
    else:
        up7 = UpSampling3D((2, 2, 2))(relu6_2)
    add7 = concatenate([up7, relu1_2], name="concatenate_3")  # up7,relu1_2
    conv7_1 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_13")(add7)
    relu7_1 = Activation('relu', name="activation_13")(conv7_1)
    conv7_2 = Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), name="conv3d_14")(relu7_1)
    relu7_2 = Activation('relu', name="activation_14")(conv7_2)

    if not feats:
        last_layer = Conv3D(labs, (1, 1, 1), name="conv3d_15")(relu7_2)
    else:
        last_layer = relu7_2
    return last_layer


def unet(input_shape, labels, arch_mode='large', loss_mode='sparse_cce', custom_loss_weights=None, loadweights=False, verbose=False):
    arch_modes = ['large', 'medium', 'small']
    loss_modes = ['scce', 'weighted_scce', 'focal', 'prob_scce', 'weighted_prob_scce']

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    backend.set_image_data_format('channels_last')
    input_shape = (*input_shape, 1)

    inpt = Input(input_shape, name="input_1")

    if arch_mode == 'large':
        last_layer = unet_large(inpt, len(labels))
    elif arch_mode == 'medium':
        last_layer = unet_medium(inpt, len(labels))
    elif arch_mode == 'small':
        last_layer = unet_small(inpt, len(labels))
    else:
        raise RuntimeError("Parameter 'arch_mode' must be one of: {}".format(arch_modes))

    if loss_mode == 'scce':
        out = Activation("softmax", name="activation_last")(last_layer)
        loss = 'sparse_categorical_crossentropy'
    elif loss_mode == 'weighted_scce':
        out = Activation("softmax", name="activation_last")(last_layer)
        loss = weighted_sparse_categorical_crossentropy(labels)
    elif loss_mode == 'focal':
        out = Activation("softmax", name="activation_last")(last_layer)
        loss = focal_loss()
    elif loss_mode == 'prob_scce':
        out = Activation("sigmoid", name="activation_last")(last_layer)
        loss = 'sparse_categorical_crossentropy'
    elif loss_mode == 'weighted_prob_scce':
        out = Activation("sigmoid", name="activation_last")(last_layer)
        loss = weighted_sparse_categorical_crossentropy(labels)
    elif loss_mode == 'custom_weighted_scce':
        if custom_loss_weights is None:
            raise RuntimeError("You need to pass 'custom_loss_weights' parameter when using 'custom_weighted_scce'.")
        out = Activation("softmax", name="activation_last")(last_layer)
        loss = custom_weighted_sparse_categorical_cross_entropy(custom_loss_weights)
    else:
        raise RuntimeError("Parameter 'loss_mode' must be one of: {}".format(loss_modes))

    non_zero_tp = NonZeroCategoricalTruePositives()

    model = Model(inputs=inpt, outputs=out)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'sparse_categorical_crossentropy',
                                                        'sparse_categorical_accuracy', non_zero_tp], run_eagerly=True)

    if loadweights:
        model.load_weights(loadweights)

    if verbose:
        model.summary(line_length=150)
        for layer in model.layers:
            print(layer.name, layer.output_shape)

    return model


class NonZeroCategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='non_zero_categorical_true_positives', **kwargs):
        super(NonZeroCategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.sizes = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.reshape(tf.cast(y_true, 'float32'), shape=(-1,1)))
        y_pred = tf.squeeze(tf.reshape(tf.cast(tf.argmax(y_pred, axis=-1), 'float32'), shape=(-1,1)))
        mask = y_true != 0
        vals = y_pred[mask] == y_true[mask]
        vals = tf.cast(vals, 'float32')
        self.true_positives.assign_add(tf.reduce_sum(vals))
        self.sizes.assign_add(tf.reduce_sum(tf.cast(mask, 'float32')))

    def result(self):
        return self.true_positives / self.sizes

    def reset_states(self):
        self.true_positives.assign(0.)
        self.sizes.assign(0.)


def weighted_sparse_categorical_crossentropy(classes, mode=None):
    def wscc(target, output):
        loss = tf.keras.backend.sparse_categorical_crossentropy(target, output)

        if mode is not None:
            ws = class_occurence(target, labs=classes, mode=mode)
        else:
            ws = class_occurence(target, labs=classes)
        t = np.copy(target)
        weights = np.zeros(target.shape, dtype=np.float32)

        for i, u in enumerate(classes):
            weights[t == u] = ws[i]

        weights = weights.reshape(loss.shape)
        return loss * weights

    return wscc


def custom_weighted_sparse_categorical_cross_entropy(weights):
    def cwscc(target, output):
        loss = tf.keras.backend.sparse_categorical_crossentropy(target, output)
        if loss.shape != weights.shape:
            raise RuntimeError("The shape of the weights {} must be the same as the "
                               "shape of the error (loss) matrix {}".format(weights.shape, loss.shape))
        return loss * weights
    return cwscc


def focal_loss(alpha=.25, gamma=2):
    def focal(target, output):
        loss = tf.keras.backend.sparse_categorical_crossentropy(target, output)

        eps = tf.constant(backend.epsilon())  # non-zero
        t = to_categorical(target, num_classes=output.shape[-1])  # from sparse to categorical
        o = output / math_ops.reduce_sum(output)  # normalise
        o = clip_ops.clip_by_value(o, eps, 1 - eps)  # clip to non-zero
        weights = math_ops.reduce_sum(alpha * np.power(1 - o, gamma) * t, axis=-1)

        return loss * weights
    return focal


class PlotMetrics(tf.keras.callbacks.Callback):
    def __init__(self, x_test=None, y_test=None, ts=False):
        super(PlotMetrics).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.ts = ts
        self.loss, self.acc, self.tp = ([], [], [])

    def on_epoch_end(self, epoch, logs=None):
        loss = 'sparse_categorical_crossentropy'
        loss_val = 'val_sparse_categorical_crossentropy'
        acc = 'sparse_categorical_accuracy'
        acc_val = 'val_sparse_categorical_accuracy'
        tp = 'non_zero_categorical_true_positives'
        tp_val = 'val_non_zero_categorical_true_positives'

        # if self.x_test is not None and self.y_test is not None:
        #     ev = dict(zip(self.model.metrics_names, self.model.evaluate(self.x_test, self.y_test, verbose=0)))
        #     loss_tst, acc_tst, tp_tst = (ev[loss], ev[acc], ev[tp])
        #     self.loss.append(loss_tst)
        #     self.acc.append(acc_tst)
        #     self.tp.append(tp_tst)
        #     loss_tst, acc_tst, tp_tst = (self.loss, self.acc, self.tp)
        # else:
        #     loss_tst, acc_tst, tp_tst = (None, None, None)

        if epoch >= 1:
            x = np.arange(0, epoch+1)+1

            loss_t = [*self.model.history.history[loss], logs.get(loss)]
            loss_v = [*self.model.history.history[loss_val], logs.get(loss_val)]

            acc_t = [*self.model.history.history[acc], logs.get(acc)]
            acc_v = [*self.model.history.history[acc_val], logs.get(acc_val)]
            y_max = len(acc_v) - np.argmax(acc_v[::-1])
            plot_metric(x, loss_t, loss_v, acc_t, acc_v,
                        #loss_tst=loss_tst, acc_tst=acc_tst,
                        title="Training and Validation Loss and Accuracy | Best: epoch {}".format(y_max),
                        save='accuracy.png', colour_scheme='blues', ts=self.ts)

            tp_t = [*self.model.history.history[tp], logs.get(tp)]
            tp_v = [*self.model.history.history[tp_val], logs.get(tp_val)]
            y_max = len(tp_v) - np.argmax(tp_v[::-1])
            plot_metric(x, loss_t, loss_v, tp_t, tp_v,
                        #loss_tst=loss_tst, acc_tst=tp_tst,
                        title="Training and Validation Loss and True Positives | Best: epoch {}".format(y_max),
                        save='tp.png', colour_scheme='greens', mname='Non-zero TP', ts=self.ts)


class PlotConfusion(tf.keras.callbacks.Callback):
    def __init__(self, labels, x_val, y_val, x_test=None, y_test=None, x_train=None, y_train=None, maps=None, ts=False):
        super(PlotConfusion).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.labels = labels
        self.sample = maps
        self.ts = ts

    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = np.argmax(self.model.predict(self.x_val), axis=-1)
        confusion_matrix(self.y_val, y_pred_val.ravel(), self.labels, save='epoch{:04d}_val.png'.format(epoch+1),
                         title="Validation Confusion Matrix | Epoch: {}".format(epoch+1), ts=self.ts)
        if self.x_test is not None and self.y_test is not None:
            y_pred_test = np.argmax(self.model.predict(self.x_test), axis=-1).ravel()
            confusion_matrix(self.y_test, y_pred_test, self.labels, save='epoch{:04d}_tst.png'.format(epoch+1),
                             title="Testing Confusion Matrix | Epoch: {}".format(epoch+1), ts=self.ts)
        if self.x_train is not None and self.y_train is not None:
            y_pred_train = np.argmax(self.model.predict(self.x_train), axis=-1).ravel()
            confusion_matrix(self.y_train, y_pred_train, self.labels, save='epoch{:04d}_trn.png'.format(epoch+1),
                             title="Training Confusion Matrix | Epoch: {}".format(epoch+1), ts=self.ts)

        if self.sample is not None:
            if self.sample.has_tiles:
                    im = y_pred_val[0:self.sample.no_tiles]
            else:
                im = y_pred_val[0]
            #print(im)
            self.sample.pred = im
            self.sample.save_pred()


def train_unet(data, labels, epochs=50, batch=1, arch_mode='large', loss_mode='scce',
               patience=10, querymaps=False, verbose=False, callbacks=None):
    # TODO paralelise this
    if not str(type(data[0])) == "<class 'proteins.Sample'>":
        raise RuntimeError("Data must be of type protein.Sample, not {}.".format(type(data[0])))

    data = [d for d in data]
    d_train, d_test = train_test_split(data, test_size=0.1)  # 90% / 10%
    d_train, d_val = train_test_split(d_train, test_size=0.2 / (1 - 0.1))  # 70% / 20%
    train = [(d.res, d.id) for d in d_train]
    val = [(d.res, d.id) for d in d_val]
    test = [(d.res, d.id) for d in d_test]
    print("Validating:", val)
    print("Testing:", test)
    print("\n-------------------------- DATA SPLIT:\ntrain: {}\n  val: {}\n test: {}".format(
        len(train), len(val), len(test)))

    class_occurence(d_train, mode='props', per_sample=querymaps, save=True, file_appendix='train', verbose=True)
    class_occurence(d_val, mode='props', per_sample=querymaps, save=True, file_appendix='val', verbose=True)
    class_occurence(d_test, mode='props', per_sample=querymaps, save=True, file_appendix='test', verbose=True)

    if loss_mode == 'custom_weighted_scce':
        x_train, x_val, x_test,\
            y_train, y_val, y_test,\
            w_train = reshape_data(d_train, d_val, d_test, weights=True)
    else:
        x_train, x_val, x_test, y_train, y_val, y_test = reshape_data(d_train, d_val, d_test)
        w_train = None

    mshape = str(data[0].map.shape[0])+'x'+str(data[0].map.shape[1])+'x'+str(data[0].map.shape[2])
    mname = "model_{}".format(mshape)
    model = unet(x_train.shape[1:-1], labels, arch_mode=arch_mode, loss_mode=loss_mode,
                 custom_loss_weights=w_train, verbose=verbose)

    if callbacks is None:
        callbacks = ['checkpoints', 'metrics', 'confusion', 'maps', 'stopping', 'tests']  #, 'trains', 'timestamp']
    cbs = []
    if 'checkpoints' in callbacks:
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join('checkpoints', mname + '_epoch{epoch:04d}_checkpoint.h5'), save_weights_only=False))
    if 'stopping' in callbacks:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor='val_non_zero_categorical_true_positives',
                                                    patience=patience))
    if 'metrics' in callbacks:
        tests = (x_test, y_test) if 'tests' in callbacks else (None, None)
        ts = 'timestamp' in callbacks
        cbs.append(PlotMetrics(x_test=tests[0], y_test=tests[1], ts=ts))
    if 'confusion' in callbacks:
        maps = d_val[0] if 'maps' in callbacks else None
        tests = (x_test, y_test) if 'tests' in callbacks else (None, None)
        trains = (x_train, y_train) if 'trains' in callbacks else (None, None)
        ts = 'timestamp' in callbacks
        cbs.append(PlotConfusion(labels, x_val, y_val, x_test=tests[0], y_test=tests[1],
                                 x_train=trains[0], y_train=trains[1], maps=maps, ts=ts))

    print("\n-------------------------- TRAINING BEGINS")
    print("input: {} | arch: {} | loss: {} | epochs: {} | batch: {}\n". format(
        x_train.shape[1:-1], arch_mode, loss_mode, epochs, batch))
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch,  # verbose=2,
              callbacks=cbs)

    # model.save_weights(os.path.join('checkpoints', mname+'_final_weights.h5'))
    # model.save(os.path.join('checkpoints', mname + datetime.now().strftime("_%Y%m%d-%H%M%S")+'.h5'))
    model.save(os.path.join('checkpoints', mname + '_final.h5'))

    print("\n-------------------------- TRAINING ENDS")
    print()
    print("\n-------------------------- GENERATING PREDICTIONS")

    predict_unet(d_test, model_name=mname+'_final.h5')

    print("\n-------------------------- GENERATING PREDICTIONS")


def predict_unet(data, model_name="model_general_64x64x64_final.h5", batch=1, evaluate=False, querymaps=False, verbose=False):

    if not os.path.exists("checkpoints") or len(os.listdir("checkpoints")) == 0:
        raise RuntimeError("No trained models available.")

    if model_name not in os.listdir("checkpoints"):
        print("WARNING: Model {} does not exist in directory \"checkpoints\". Chosing the last saved checkpoint instead".format(model_name))
        model_name = os.listdir("checkpoints")[-1]

    if not str(type(data[0])) == "<class 'proteins.Sample'>":
        raise RuntimeError("Data must be of type protein.Sample, not {}.".format(type(data[0])))

    if len(data[0].map.shape) != 3:
        raise RuntimeError("Data must be 3D in and in the dimension the model was trained on.")

    if evaluate:
        class_occurence(data, mode='props', per_sample=querymaps, save=True, file_appendix='pred_eval', verbose=True)
        # ydata = np.expand_dims([d.lab for d in data], axis=-1)
        xdata, ydata = reshape_data(data)
    # xdata = np.expand_dims([d.map for d in data], axis=-1)
    else:
        xdata = reshape_data(data, eval=False)

    model = tf.keras.models.load_model(os.path.join("checkpoints", model_name))
    #                                       custom_objects={'wscc': weighted_sparse_categorical_crossentropy(ydata)})

    mshape = model.layers[0]._batch_input_shape[1:-1]

    if mshape != xdata[0].shape[:-1]:
        raise RuntimeError("This model was trained on images with following dimensions: {}. You are attempting to "
                           "predict images of dimension {}. Please make sure you preprocess your data to an "
                           "appropriate shape.\n\nThis is likely to be a result of 'scale' pre-processing, try using"
                           " --cshape flag along with 'scale' preprocessing option to specify the desired shape of your"
                           " data.".format(mshape, data[0].map.shape))

    preds = model.predict(xdata, batch_size=batch)
    preds = tf.argmax(preds, axis=-1).numpy()
    if evaluate:
        model.evaluate(xdata, ydata, batch_size=batch)

    step = 0
    for i, p in enumerate(data):
        data[i].pred = preds[step:step+data[i].no_tiles]
        data[i].save_pred()
        if evaluate:
            data[i].save_diff()
        step += data[i].no_tiles

    # for i, p in enumerate(preds):
    #     data[i].pred = tf.argmax(p, axis=-1).numpy()
    #     data[i].save_pred()
    #     if evaluate:
    #         data[i].save_diff()


def reshape_data(d_train, d_val=None, d_test=None, eval=True, weights=False):
    """Reshape Sample data for training in tensorflow. Will return tiles is they exist instead of map."""

    if not d_train[0].has_tiles:
        x_train = np.expand_dims([d.map for d in d_train], axis=-1)
        if eval:
            y_train = np.expand_dims([d.lab for d in d_train], axis=-1)
        if d_val is not None:
            x_val = np.expand_dims([d.map for d in d_val], axis=-1)
            if eval:
                y_val = np.expand_dims([d.lab for d in d_val], axis=-1)
        if d_test is not None:
            x_test = np.expand_dims([d.map for d in d_test], axis=-1)
            if eval:
                y_test = np.expand_dims([d.lab for d in d_test], axis=-1)
        if weights:
            w_train = np.asarray([d.weight for d in d_train])
    else:
        x_train = np.expand_dims([tile for d in d_train for tile in d.tiles], axis=-1)
        x_train = x_train.reshape(-1, *x_train.shape[-4:])
        if eval:
            y_train = np.expand_dims([tile for d in d_train for tile in d.ltiles], axis=-1)
            y_train = y_train.reshape(-1, *y_train.shape[-4:])
        if d_val is not None:
            x_val = np.expand_dims([tile for d in d_val for tile in d.tiles], axis=-1)
            x_val = x_val.reshape(-1, *x_val.shape[-4:])
            if eval:
                y_val = np.expand_dims([tile for d in d_val for tile in d.ltiles], axis=-1)
                y_val = y_val.reshape(-1, *y_val.shape[-4:])
        if d_test is not None:
            x_test = np.expand_dims([tile for d in d_test for tile in d.tiles], axis=-1)
            x_test = x_test.reshape(-1, *x_test.shape[-4:])
            if eval:
                y_test = np.expand_dims([tile for d in d_test for tile in d.ltiles], axis=-1)
                y_test = y_test.reshape(-1, *y_test.shape[-4:])
        if weights:
            w_train = np.asarray([tile for d in d_train for tile in d.wtiles])
            w_train = w_train.reshape(-1, *w_train.shape[-4:])

    ret = [x_train]
    if d_val:
        ret.append(x_val)
    if d_test:
        ret.append(x_test)
    if eval:
        ret.append(y_train)
        ret.append(y_val)
        ret.append(y_test)
    if weights:
        ret.append(w_train)
    return tuple(ret)


def patch_nd(img, scale=3, ravel=True, zeros=False):
    """
    Default method for feature extraction. Simple n-dimensional dense patch extraction.
    """

    feats = []
    shape = img.shape

    for n in range(img.size):

        # work out the index of current pixel/voxel; base-n conversion
        indices = []
        prev = n
        for i in reversed(range(len(shape))):
            indices.append(int(prev % img.shape[i]))
            prev = prev / img.shape[i]

        # work out shapes of slices
        slices = []
        for i, ind in enumerate(reversed(indices)):
            # from array of starts and stops for x y and z
            start = ind  # index of cropped array minus half length of patch
            stop = ind + scale  # index of cropped array minus half length of patch plus patch length
            slices.append(slice(start, stop))

        # slice window of size PATCH from index position
        feature = img[slices]
        if feature.size != int(math.pow(scale, len(shape))):
            # reject edge features
            continue
        if ravel:
            feature = np.asarray(feature.ravel())
        feats.append(feature)

    feats = np.asarray(feats)

    if feats.shape[0] != np.prod((np.array(shape) - (scale // 2 * 2))):
        # print(feats.shape)
        # print(np.prod((np.array(shape) - (scale // 2 * 2))))
        # check the size of the final feature space
        raise RuntimeError("Error in feature extraction. Check if patch_size is an odd number!")

    if zeros:
        # TODO refactor this so that it is placed in the matrix of np.zeros((*shape, f_dim))
        margin = scale // 2
        f_dim = feats.shape[-1]
        new_s = shape + (f_dim,)
        feats_s = tuple(np.array(shape) - (scale // 2 * 2)) + (f_dim,)

        new_feats = np.zeros(new_s, dtype=feats.dtype)
        feats = feats.reshape(feats_s)

        crop = []
        for d in range(len(shape)):
            crop.append(slice(margin, -margin))

        new_feats[crop] = feats
        new_feats = new_feats.reshape(np.prod(new_feats.shape[:-1]), f_dim)
        return new_feats
    return np.asarray(feats)
