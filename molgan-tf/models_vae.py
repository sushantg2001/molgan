# %%
# models...vae.py
import os
import traceback
import random
import pickle
import numpy as np
import rdkit
from rdkit import Chem
import tensorflow as tf
import pickle
import gzip
import random
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import pandas as pd
import math
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from collections import defaultdict
import pprint

class GraphVAEModel:
    def __init__(self, vertexes, edges, nodes, features, embedding_dim, encoder_units, decoder_units, variational,
                 encoder, decoder, soft_gumbel_softmax=False, hard_gumbel_softmax=False, with_features=True):

        self.vertexes, self.nodes, self.edges, self.embedding_dim, self.encoder, self.decoder = \
            vertexes, nodes, edges, embedding_dim, encoder, decoder

        self.training = tf.placeholder_with_default(False, shape=())
        self.variational = tf.placeholder_with_default(variational, shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(
            soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(
            hard_gumbel_softmax, shape=())
        self.temperature  # %%
# vae.py..optimiser


class GraphVAEOptimizer(object):

    def __init__(self, model, learning_rate=1e-3):
        self.kl_weight = tf.placeholder_with_default(1., shape=())
        self.la = tf.placeholder_with_default(1., shape=())

        edges_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.edges_labels,
                                                            logits=model.edges_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.edges_loss = tf.reduce_sum(edges_loss, [-2, -1])

        nodes_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.nodes_labels,
                                                            logits=model.nodes_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.nodes_loss = tf.reduce_sum(nodes_loss, -1)

        self.loss_ = self.edges_loss + self.nodes_loss
        self.reconstruction_loss = tf.reduce_mean(self.loss_)

        self.p_z = tf.distributions.Normal(tf.zeros_like(model.embeddings_mean),
                                           tf.ones_like(model.embeddings_std))
        self.kl = tf.reduce_mean(tf.reduce_sum(
            tf.distributions.kl_divergence(model.q_z, self.p_z), axis=-1))

        self.ELBO = - self.reconstruction_loss - self.kl

        self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + \
            (model.value_logits_fake - model.rewardF) ** 2

        self.loss_RL = - model.value_logits_fake

        self.loss_RL = - model.value_logits_fake

        self.loss_VAE = tf.cond(model.variational,
                                lambda: self.reconstruction_loss + self.kl_weight * self.kl,
                                lambda: self.reconstruction_loss)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        self.loss_RL *= tf.abs(tf.stop_gradient(self.loss_VAE / self.loss_RL))

        self.VAE_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_VAE = self.VAE_optim.minimize(

            loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_VAE, lambda: 0.) + tf.cond(
                tf.less(self.la, 1), lambda: (1 - self.la) * self.loss_RL, lambda: 0.),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder'))

        self.V_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_V = self.V_optim.minimize(
            loss=self.loss_V,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))

        self.log_likelihood = self.__log_likelihood
        self.model = model

    def __log_likelihood(self, n):

        z = self.model.q_z.sample(n)

        log_p_z = self.p_z.log_prob(z)
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

        log_p_x_z = -self.loss_

        log_q_z_x = self.model.q_z.log_prob(z)
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

        print([a.shape for a in (log_p_z, log_p_x_z, log_q_z_x)])

        return tf.reduce_mean(tf.reduce_logsumexp(
            tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))
 = tf.placeholder_with_default(1., shape=())

        self.edges_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes, vertexes))
        self.nodes_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes))
        self.node_features = tf.placeholder(
            dtype=tf.float32, shape=(None, vertexes, features))

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.adjacency_tensor = tf.one_hot(
            self.edges_labels, depth=edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(
            self.nodes_labels, depth=nodes, dtype=tf.float32)

        with tf.variable_scope('encoder'):
            outputs = self.encoder(
                (self.adjacency_tensor,
                 self.node_features if with_features else None, self.node_tensor),
                units=encoder_units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=encoder_units[-1], activation=tf.nn.tanh,
                                         training=self.training, dropout_rate=0.)

            self.embeddings_mean = tf.layers.dense(
                outputs, embedding_dim, activation=None)
            self.embeddings_std = tf.layers.dense(
                outputs, embedding_dim, activation=tf.nn.softplus)
            self.q_z = tf.distributions.Normal(
                self.embeddings_mean, self.embeddings_std)

            self.embeddings = tf.cond(self.variational,
                                      lambda: self.q_z.sample(),
                                      lambda: self.embeddings_mean)

        with tf.variable_scope('decoder'):
            self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                                                                training=self.training, dropout_rate=0.)

        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
                (self.edges_argmax, self.nodes_argmax), \
                (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
                (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
                (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                (self.edges_logits, self.nodes_logits), temperature=self.temperature)

            self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                                     default=lambda: self.edges_softmax,
                                     exclusive=True)

            self.nodes_hat = tf.case({self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                                     default=lambda: self.nodes_softmax,
                                     exclusive=True)

        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.V_x(
                (self.adjacency_tensor, None, self.node_tensor), units=encoder_units)

        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.V_x(
                (self.edges_hat, None, self.nodes_hat), units=encoder_units)

    def V_x(self, inputs, units):
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.encoder(
                inputs, units=units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=0.)

            outputs = tf.layers.dense(
                outputs, units=1, activation=tf.nn.sigmoid)

        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))
