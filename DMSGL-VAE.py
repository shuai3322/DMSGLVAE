
# python3.9 test

def load_data(train_path, valid_path, test_path):
    import pandas as pd
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle

    data = pd.read_excel(train_path)
    data = shuffle(data, random_state=1234)
    scale = StandardScaler()
    data_x = data.drop(['label'], axis=1)
    feature_name = data_x.columns.values
    y = data.loc[:, ['label']]
    x = scale.fit_transform(data_x)
    x, y = shuffle(x, y, random_state=1234)
    valid_data = pd.read_excel(valid_path)
    valid_data = shuffle(valid_data, random_state=1234)

    valid_x = valid_data.drop(['label'], axis=1)
    valid_x = scale.transform(valid_x)

    valid_y = valid_data.loc[:, ['label']]
    test_data = pd.read_excel(test_path)
    test_data = shuffle(test_data, random_state=1234)
    test_x = test_data.drop(['label'], axis=1)
    test_x = scale.transform(test_x)

    test_y = test_data.loc[:, ['label']]
    return x, y, valid_x, valid_y, test_x, test_y, feature_name


import pandas as pd
import numpy as np

blood_x, blood_y, blood_valid_x, blood_valid_y,blood_test_x,blood_test_y,blood_features = load_data('b_train1_212.xlsx','b_valid1_56.xlsx','b_test1_26.xlsx')
urine_x, urine_y, urine_valid_x, urine_valid_y,urine_test_x,urine_test_y,urine_features = load_data('u_train1_212.xlsx','u_valid1_56.xlsx','u_test1_26.xlsx')

from sklearn.metrics import *
import tensorflow as tf
from keras.layers import *
from keras import Model, Input
from keras.layers import Layer
from keras import backend as K
from keras.losses import mean_squared_error


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class DMSGLVAE(Model):
    def __init__(self, inp_shape, latent_dim,d_dim, **kwargs):
        super(DMSGLVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.inp_shape = inp_shape
        self.d_dim=d_dim

        self.blood_encoder = self.build_encoder_network(name='blood')
        self.urine_encoder = self.build_encoder_network(name='urine')

        self.blood_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='blood')
        self.urine_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='urine')

        self.grade_classifier = self.build_classification_network()

        self.blood_encoder.summary()
        self.blood_decoder.summary()
        self.grade_classifier.summary()

        self.MMDR_model = self.build_DMSGLVAE()


        self.MMDR_model.summary()


    def build_DMSGLVAE(self):

        MMDR_input_blood = Input(shape=self.inp_shape, name='MMDR_input_blood')
        MMDR_input_urine = Input(shape=self.inp_shape, name='MMDR_input_urine')

        blood_encoder_out = self.blood_encoder(MMDR_input_blood)
        urine_encoder_out = self.urine_encoder(MMDR_input_urine)

        blood_decoder_out = self.blood_decoder([blood_encoder_out[3], blood_encoder_out[4]])

        urine_blood_decoder_out = self.blood_decoder([urine_encoder_out[3], blood_encoder_out[4]])

        urine_decoder_out = self.urine_decoder([urine_encoder_out[3], urine_encoder_out[4]])

        blood_urine_decoder_out = self.urine_decoder([blood_encoder_out[3], urine_encoder_out[4]])

        common_code = Lambda(lambda x: (x[0] + x[1]) / 2)([blood_encoder_out[3], urine_encoder_out[3]])

        com_urine_decoder_out = self.urine_decoder([common_code, urine_encoder_out[4]])
        com_blood_decoder_out = self.blood_decoder([common_code, blood_encoder_out[4]])


        grade_out = self.grade_classifier([common_code,blood_encoder_out[4], urine_encoder_out[4]])

        model = Model(inputs=[MMDR_input_blood, MMDR_input_urine],
                      outputs=[blood_encoder_out, urine_encoder_out,
                               blood_decoder_out, com_blood_decoder_out,
                               urine_decoder_out, com_urine_decoder_out,grade_out], name="MMDR")
        return model


    def predict_grade(self, x):
        out = self.MMDR_model.predict(x)
        return out[-1]

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data

        with tf.GradientTape() as tape:
            blood_encoder_out, urine_encoder_out, \
            blood_decoder_out, com_blood_decoder_out, \
            urine_decoder_out, com_urine_decoder_out, grade_out = self.MMDR_model(x)

            z_mean_blood, z_log_var_blood, z_blood, common_blood, spec_blood = blood_encoder_out
            z_mean_urine, z_log_var_urine, z_urine, common_urine, spec_urine = urine_encoder_out

            reconstruction_loss = tf.reduce_mean(tf.sqrt(
                tf.keras.losses.mean_squared_error(x[0], blood_decoder_out))
            ) + tf.reduce_mean(tf.sqrt(
                tf.keras.losses.mean_squared_error(x[0], com_blood_decoder_out))
            ) + tf.reduce_mean(tf.sqrt(
                tf.keras.losses.mean_squared_error(x[1], urine_decoder_out))
            ) + tf.reduce_mean(tf.sqrt(
                tf.keras.losses.mean_squared_error(x[1], com_urine_decoder_out))
            )

            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var_blood - tf.square(z_mean_blood) - tf.exp(z_log_var_blood)) \
                      + (-0.5 * tf.reduce_mean(1 + z_log_var_urine - tf.square(z_mean_urine) - tf.exp(z_log_var_urine)))

            com_loss = tf.reduce_mean(tf.sqrt(mean_squared_error(common_blood, common_urine)))

            spec_loss = tf.reduce_mean(tf.sqrt(mean_squared_error(spec_blood, spec_urine)))

            com_spec_loss = com_loss / spec_loss

            grade_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, grade_out))

            total_loss = 3.5 * reconstruction_loss + 0.25* kl_loss + 1 * com_spec_loss + 0.3 * grade_loss  # reconstruction_loss *= 0.1

        # The training happens here.
        grads = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        print(">>>>>>>>>>>>>>metrics_names:", self.metrics_names)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, grade_out)

        return {
            "loss": total_loss,
            "rec_loss": reconstruction_loss,
            "com_loss": com_loss,
            "spec_loss": spec_loss,
            "kl_loss": kl_loss,
            "cs_loss": com_spec_loss,
            "grade_loss": grade_loss,
            self.metrics[0].name: self.metrics[0].result(),
            self.metrics[1].name: self.metrics[1].result()
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data

        blood_encoder_out, urine_encoder_out, \
        blood_decoder_out, com_blood_decoder_out, \
        urine_decoder_out, com_urine_decoder_out, grade_out = self.MMDR_model(x)

        z_mean_blood, z_log_var_blood, z_blood, common_blood, spec_blood = blood_encoder_out
        z_mean_urine, z_log_var_urine, z_urine, common_urine, spec_urine = urine_encoder_out

        reconstruction_loss = tf.reduce_mean(tf.sqrt(
            tf.keras.losses.mean_squared_error(x[0], blood_decoder_out))
        ) + tf.reduce_mean(tf.sqrt(
            tf.keras.losses.mean_squared_error(x[0], com_blood_decoder_out))
        ) + tf.reduce_mean(tf.sqrt(
            tf.keras.losses.mean_squared_error(x[1], urine_decoder_out))
        ) + tf.reduce_mean(tf.sqrt(
            tf.keras.losses.mean_squared_error(x[1], com_urine_decoder_out))
        )

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var_blood - tf.square(z_mean_blood) - tf.exp(z_log_var_blood)) \
                  + (-0.5 * tf.reduce_mean(1 + z_log_var_urine - tf.square(z_mean_urine) - tf.exp(z_log_var_urine)))

        com_loss = tf.reduce_mean(tf.sqrt(mean_squared_error(common_blood, common_urine)))

        spec_loss = tf.reduce_mean(tf.sqrt(mean_squared_error(spec_blood, spec_urine)))

        com_spec_loss = com_loss / spec_loss

        grade_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, grade_out))

        total_loss = 3 * reconstruction_loss + 0.25 * kl_loss + 1 * com_spec_loss + 0.3 * grade_loss

        self.compiled_metrics.update_state(y, grade_out)

        return {
            "loss": total_loss,
            "rec_loss": reconstruction_loss,
            "com_loss": com_loss,
            "spec_loss": spec_loss,
            "kl_loss": kl_loss,
            "cs_loss": com_spec_loss,
            "grade_loss": grade_loss,
            self.metrics[0].name: self.metrics[0].result(),
            self.metrics[1].name: self.metrics[1].result()
        }

    def build_encoder_network(self, disentangled=True, name=''):
        """
        create encoder network
        :param input_shape:
        :param disentangled:
        :param name:
        :return:
        """
        input = Input(shape=self.inp_shape)
        x = Dense(256, activation='relu')(input)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        if disentangled:
            mean = Dense(self.latent_dim)(x)
            log_var = Dense(self.latent_dim)(x)
            z = Sampling()([mean, log_var])
            z1 = tf.expand_dims(z, axis=2)
            z1 = Conv1D(4,1,padding='valid',strides=1)(z1)
            s = Conv1D(1,3, padding='valid', strides=1)(z1)
            s = MaxPooling1D(pool_size=7, strides=1, padding='valid')(s)
            spec = tf.squeeze(s, -1)
            c = Conv1D(1,7, padding='valid', strides=1)(z1)
            c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
            common = tf.squeeze(c, -1)
            return Model(inputs=input, outputs=[mean, log_var, z, common, spec], name='encoder_{}'.format(name))
        else:
            return Model(inputs=input, outputs=x, name='encoder_{}'.format(name))



    def build_decoder_network(self, output_shape, name=''):

        common_input = Input(shape=(self.d_dim,), name='input_1_{}'.format(name))
        spec_input = Input(shape=(self.d_dim,), name='input_2_{}'.format(name))
        inp = Concatenate(axis=-1)([common_input, spec_input])
        l = BatchNormalization()(inp)
        l = Dense(128, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(256, activation='relu')(l)
        l = Dense(output_shape, activation='tanh')(l)
        return Model(inputs=[common_input, spec_input], outputs=l, name='decoder_{}'.format(name))

    def build_classification_network(self):

        inp_comm = Input((self.d_dim,), name='common_input')
        inp_blood = Input((self.d_dim,), name='blood_input')
        inp_urine = Input((self.d_dim,), name='urine_input')
        fusion_feature = Concatenate(axis=-1)([inp_comm,inp_blood, inp_urine])

        x = Dense(32, activation='relu')(fusion_feature)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid', name='grading')(x)
        model = Model(inputs=[inp_comm,inp_blood, inp_urine], outputs=x)
        return model


from keras.optimizers import adam_v2
tf.compat.v1.disable_eager_execution()
print(tf.__version__)
def get_flops_params():
     sess = tf.compat.v1.Session()
     graph = sess.graph
     flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
     params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
     print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
model = DMSGLVAE(trains[0].shape[1:], latent_dim=48,d_dim=40) # latent_dimatent=48
METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc'),
    ]
model.compile(optimizer=adam_v2.Adam(lr=0.0001), metrics=METRICS) # 损失函数
# get_flops_params()
# model.summary()
from keras import callbacks
from keras.callbacks import ModelCheckpoint
filepath = r'models/best_model.h5'
checkpoint = callbacks.ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only='True',
                             save_weights_only='True',
                             mode='max',
                        period=1)

lrreduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0) # 回调函数

import time
import os
fit_start = time.perf_counter()
history = model.fit(trains, trains_label,batch_size=8,epochs=500, shuffle=False,validation_data=(valids, valids_label), callbacks=[checkpoint])
fit_end = time.perf_counter()
print("train time is: ", fit_end-fit_start)
print(f'准确率：{np.mean(history.history["acc"])}')
print(f'AUC：{np.mean(history.history["auc"])}')
print(f'准确率：{np.mean(history.history["val_acc"])}')
print(f'AUC：{np.mean(history.history["val_auc"])}')

model_path = r'models/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_weights(os.path.join(model_path, 'DMSGLVAE_weights'))
print("Save model!")

model.load_weights(r'DMSGLVAE_weights')

