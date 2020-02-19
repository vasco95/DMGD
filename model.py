import numpy as np
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, config):
        # This is content branch. Everything related to structure is content.
        # Except for Homophily. The neighbors are based on graphs.
        self.struc_size = config['struc_size']
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        self.learning_rate = config['learning_rate']
        self.num_clusters = config['num_clusters']
        self.hid_dim = self.encoder[-1]

    def _add_placeholders(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.struc_size], name = "input_x")

        # Homophily neighbors for structure
        self.input_x_neigh1 = tf.placeholder(tf.float32, [None, self.struc_size], name = "input_neigh1")
        self.input_x_neigh2 = tf.placeholder(tf.float32, [None, self.struc_size], name = "input_neigh2")

        # Dual Inputs
        self.input_alpha = tf.placeholder(tf.float32, [None], name = "input_alpha")
        self.input_dual_neigh1 = tf.placeholder(tf.float32, [None, self.struc_size], name = "input_dual_neigh1")
        self.input_dual_neigh2 = tf.placeholder(tf.float32, [None, self.struc_size], name = "input_dual_neigh2")
        self.input_mul1 = tf.placeholder(tf.float32, [None], name = "input_mul1")
        self.input_mul2 = tf.placeholder(tf.float32, [None], name = "input_mul2")

        # Loss function weightage
        self.alpha = 1.0 # Recon
        self.beta = 1e2 # Homo
        self.gamma = 1.0 # Dual

    def _add_encoder(self, batch_x, reuse = False):
        xvec =  batch_x
        with tf.variable_scope("struc_encoder", reuse = reuse):
            for ii in range(len(self.encoder)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.encoder[ii], # kernel_regularizer = self.regularizer,
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)

        # struc_embeddings
        return xvec

    def _add_decoder(self, hidden_x):
        xvec = hidden_x
        with tf.variable_scope("struc_decoder"):
            for ii in range(len(self.decoder)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.decoder[ii], # kernel_regularizer = self.regularizer,
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)
            input_rec = tf.layers.dense(xvec, self.struc_size,
                                        activation=tf.nn.relu, use_bias = False, name = "struc_final_layer")

        return input_rec

    # calculate loss
    def _add_loss(self, batch_x,
                        decoded_x,
                        struc_hid,
                        struct_neigh1, struct_neigh2,
                        dual_neigh1, dual_neigh2):
        with tf.variable_scope('loss'):
            #  Loss 1 struct
            self.loss1 = tf.reduce_sum(tf.square((5.0 * batch_x + 1e-2) - decoded_x), axis=1)
            # self.loss1 = tf.reduce_sum(tf.square(batch_x - decoded_x), axis=1)

            # Homophily regularizer for Structure
            self.loss2 = tf.reduce_sum(tf.square(struc_hid - struct_neigh1), axis = 1) +\
                            tf.reduce_sum(tf.square(struc_hid - struct_neigh2), axis = 1)

            # Dual Loss
            term1 = self.input_alpha * tf.reduce_sum(tf.square(struc_hid), axis = 1)
            term2 = self.input_mul1 * tf.reduce_sum(tf.multiply(struc_hid, dual_neigh1), axis = 1) +\
                    self.input_mul2 * tf.reduce_sum(tf.multiply(struc_hid, dual_neigh2), axis = 1)
            self.loss3 = term1 - term2
            self.loss_dual = self.gamma * tf.reduce_mean(self.loss3)

            # L2 Regularization
            # reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss_auto = self.alpha * self.loss1 +\
                                self.beta * self.loss2 +\
                                self.gamma * self.loss3
                                # reg_loss
            self.loss = tf.reduce_mean(self.loss_auto)

            tf.summary.scalar("Total_Loss", self.loss)
            tf.summary.scalar("Reconstruction_Loss", self.alpha * tf.reduce_mean(self.loss1))
            tf.summary.scalar("Homophily_Loss", self.beta * tf.reduce_mean(self.loss2))
            tf.summary.scalar("Dual_Loss", self.gamma * tf.reduce_mean(self.loss3))
            # tf.summary.scalar("L2_Reg_Loss", reg_loss)

    def create_network(self):
        self._add_placeholders()

        self.initializer = tf.contrib.layers.xavier_initializer(uniform = False)
        self.lr = tf.train.inverse_time_decay(0.03, 60, decay_steps=1, decay_rate=0.9999)

        # Define L2-regularizer
        # self.regularizer = tf.contrib.layers.l2_regularizer(1e-5)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(0.0, 0.0)
        # tf.set_random_seed(10)

        self.struc_hid = self._add_encoder(self.input_x)
        self.struct_neigh1 = self._add_encoder(self.input_x_neigh1, reuse = True)
        self.struct_neigh2 = self._add_encoder(self.input_x_neigh2, reuse = True)
        self.dual_neigh1 = self._add_encoder(self.input_dual_neigh1, reuse = True)
        self.dual_neigh2 = self._add_encoder(self.input_dual_neigh2, reuse = True)
        self.decoded_x = self._add_decoder(self.struc_hid)

        self._add_loss(self.input_x, self.decoded_x, self.struc_hid,
                            self.struct_neigh1, self.struct_neigh2,
                                self.dual_neigh1, self.dual_neigh2)

    def initialize_summary_writer(self, sess, fname):
        self.all_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(fname, sess.graph)

    def initialize_optimizer(self, config):
        # Optimizer 1 for pretraining
        self.global_step_1 = tf.Variable(0, name = "global_step_1", trainable = False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_1 = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars_1 = optimizer_1.compute_gradients(self.loss)
            self.train_op_1 = optimizer_1.apply_gradients(self.grads_and_vars_1, global_step = self.global_step_1)

        # Optimizer 2 for Dual loss
        self.global_step_2 = tf.Variable(0, name = "global_step_2", trainable = False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_2 = tf.train.AdamOptimizer(1e-3)
            self.grads_and_vars_2 = optimizer_2.compute_gradients(self.loss_dual)
            self.train_op_2 = optimizer_2.apply_gradients(self.grads_and_vars_2, global_step = self.global_step_2)
        self.combine_opt = tf.group(self.train_op_1, self.train_op_2)
        # self.combine_opt = self.train_op_2

    def train_step(self, sess, feed_dict, print_this = True):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]
        feed[self.input_x_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_x_neigh2] = feed_dict["struc_input_neigh2"]
        # Dual Inputs: Dummy values for summary only
        batch_size = feed_dict["struc_input"].shape[0]
        feed[self.input_alpha] = np.zeros(batch_size)
        feed[self.input_dual_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_dual_neigh2] = feed_dict["struc_input_neigh1"]
        feed[self.input_mul1] = np.zeros(batch_size)
        feed[self.input_mul2] = np.zeros(batch_size)

        run_vars = [self.train_op_1, self.global_step_1, self.loss, self.all_summary]
        _, idx, rloss, summ = sess.run(run_vars, feed_dict = feed)

        self.writer.add_summary(summ, idx)
        if print_this:
            print(idx, 'LOSS =', rloss)

    def train_step_dual(self, sess, feed_dict, mode = "all", print_this = True):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]
        feed[self.input_x_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_x_neigh2] = feed_dict["struc_input_neigh2"]
        # Dual Inputs
        feed[self.input_alpha] = feed_dict["input_alpha"]
        feed[self.input_dual_neigh1] = feed_dict["dual_input_neigh1"]
        feed[self.input_dual_neigh2] = feed_dict["dual_input_neigh2"]
        feed[self.input_mul1] = feed_dict["input_mul1"]
        feed[self.input_mul2] = feed_dict["input_mul2"]

        # Run both optimizers
        if mode == "all":
            # run_vars = [self.combine_opt, self.global_step_1, self.loss, self.loss_dual, self.all_summary]
            run_vars = [self.train_op_1, self.global_step_1, self.loss, self.loss_dual, self.all_summary]
        elif mode == "dual":
            run_vars = [self.train_op_2, self.global_step_1, self.loss, self.loss_dual, self.all_summary]

        _, idx, rloss, sloss, summ = sess.run(run_vars, feed_dict = feed)

        self.writer.add_summary(summ, idx)
        if print_this:
            print('Iter {}: rloss = {}, dloss = {}.'.format(idx, rloss, sloss))

    def get_hidden(self, sess, x_batch):
        feed = {}
        feed[self.input_x] = x_batch

        struc_emb = sess.run(self.struc_hid, feed_dict = feed)

        return struc_emb

    def get_decoded(self, sess, feed_dict):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]

        run_vars = [self.decoded_x]
        recon_X = sess.run(run_vars, feed_dict = feed)

        return recon_X

    def get_losses(self, sess, feed_dict):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]
        feed[self.input_x_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_x_neigh2] = feed_dict["struc_input_neigh2"]
        # Dual Inputs
        feed[self.input_alpha] = feed_dict["input_alpha"]
        feed[self.input_dual_neigh1] = feed_dict["dual_input_neigh1"]
        feed[self.input_dual_neigh2] = feed_dict["dual_input_neigh2"]
        feed[self.input_mul1] = feed_dict["input_mul1"]
        feed[self.input_mul2] = feed_dict["input_mul2"]
        # # Dual Inputs: Dummy values for summary only
        # batch_size = feed_dict["struc_input"].shape[0]
        # feed[self.input_alpha] = np.zeros(batch_size)
        # feed[self.input_dual_neigh1] = feed_dict["struc_input_neigh1"]
        # feed[self.input_dual_neigh2] = feed_dict["struc_input_neigh1"]
        # feed[self.input_mul1] = np.zeros(batch_size)
        # feed[self.input_mul2] = np.zeros(batch_size)

        # Run both optimizers
        run_vars = [self.loss_auto, self.loss3]
        l1, l2 = sess.run(run_vars, feed_dict = feed)

        return l1, l2
