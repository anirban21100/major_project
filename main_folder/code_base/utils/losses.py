import tensorflow as tf
import math
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Input, Dropout, Softmax
import tensorflow.keras.backend as K


class ArcFace(tf.keras.layers.Layer):
    def __init__(
        self,
        n_classes,
        s=30,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
        regularizer=None,
        **kwargs
    ):

        super(ArcFace, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m
        self.regularizer = regularizer

    def build(self, input_shape):

        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer="glorot_uniform",
            dtype="float32",
            trainable=True,
            regularizer=self.regularizer,
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "ls_eps": self.ls_eps,
                "easy_margin": self.easy_margin,
                "regularizer": self.regularizer,
            }
        )
        return config

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1), tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=cosine.dtype)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SphereFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):

        self.W = self.add_weight(
            name="W",
            shape=(input_shape[0][-1], self.n_classes),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self.regularizer,
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "regularizer": self.regularizer,
            }
        )
        return config

    def call(self, inputs):
        x, y = inputs
        # c = K.shape(x)[-1]
        y = tf.cast(y, dtype=tf.int32)
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        y = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=logits.dtype)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        return logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class CosFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[0][-1], self.n_classes),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self.regularizer,
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "regularizer": self.regularizer,
            }
        )
        return config

    def call(self, inputs):
        x, y = inputs
        # c = K.shape(x)[-1]
        y = tf.cast(y, dtype=tf.int32)
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        y = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=logits.dtype)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        return logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class CurricularFace(tf.keras.layers.Layer):
    def __init__(self, n_classes, m=0.5, s=64.0, regularizer=None, **kwargs):
        super(CurricularFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.regularizer = regularizer
        self.t = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Zeros(), trainable=False
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "regularizer": self.regularizer,
            }
        )
        return config

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer="glorot_uniform",
            dtype="float32",
            trainable=True,
            regularizer=self.regularizer,
        )

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cos = tf.linalg.matmul(
            tf.nn.l2_normalize(X, axis=1), tf.nn.l2_normalize(self.W, axis=0)
        )
        # cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)  # Numerical stability

        # target_logit = tf.gather_nd(cos_theta, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1))
        # target_logit = tf.expand_dims(target_logit, axis=1)

        sin_theta = tf.sqrt(1.0 - tf.square(cos))
        phi = cos * self.cos_m - sin_theta * self.sin_m

        mask = cos > phi
        phi = tf.where(cos > self.threshold, phi, cos - self.mm)

        hard_example = tf.boolean_mask(cos, mask)
        self.t = 0.01 * tf.reduce_mean(cos) + (1 - 0.01) * self.t
        cos = tf.tensor_scatter_nd_update(
            cos, tf.where(mask), hard_example * (self.t + hard_example)
        )

        one_hot = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=cos.dtype)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)

        output *= self.s
        return output
