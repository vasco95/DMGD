import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf

# np.random.seed(10)
# Batch iterator for pretraining
def batch_iter_1(in_x, in_y, batch_size, num_epochs, shuffle = True):
    data_x = np.array(in_x)
    data_y = np.array(in_y)

    data_size = data_x.shape[0]
    sample_idx = np.arange(data_size)
    order = np.arange(data_size)
    num_batches = int((data_size - 1) / batch_size) + 1

    for epoch in tqdm(range(num_epochs)):
        if shuffle:
            np.random.shuffle(order)
            data_x = data_x[order]
            data_y = data_y[order]

        idx1, idx2 = [], []
        # For feeding tne Homophily inputs
        for idx in range(data_size):
            # try:
            if np.sum(data_x[idx]) != 0.0:
                p = data_x[idx] / np.sum(data_x[idx])
                samples = np.random.choice(sample_idx, size = 2, p = p / np.sum(p))
                idx1.append(samples[0])
                idx2.append(samples[1])

            # except ZeroDivisionError:
            else:
                # print 'Exception encountered'
                idx1.append(order[idx])
                idx2.append(order[idx])
                # pass

        for batch_num in range(num_batches):
            feed_dict = {}
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            feed_dict["struc_input"] = data_y[start_index:end_index]
            feed_dict["struc_input_neigh1"] = in_y[idx1[start_index : end_index]]
            feed_dict["struc_input_neigh2"] = in_y[idx2[start_index : end_index]]

            yield feed_dict, batch_num == num_batches - 1

# Batch iterator for Dual
def batch_iter_2(in_x, in_y, in_alpha, in_theta, batch_size, num_epochs, shuffle = True, display = False):
    data_x = np.array(in_x)
    data_y = np.array(in_y)
    data_alpha = np.array(in_alpha) # N x 1
    data_theta = np.array(in_theta) # N x k
    index = np.arange(len(in_x))

    data_size = data_x.shape[0]
    sample_idx = np.arange(data_size)
    order = np.arange(data_size)
    num_batches = int((data_size - 1) / batch_size) + 1

    #######
    ### Experiment with ground truth communities
    tmp = data_alpha.reshape(-1, 1)
    mat1 = np.matmul(tmp, tmp.T)
    mat2 = np.matmul(data_theta, data_theta.T)
    dual_probs = mat1 * mat2
    print mat1.shape, mat2.shape, dual_probs.shape
    #######

    # Non zero alphas probs
    alpha_probs = []
    thresh = 1e-6
    for ii in range(data_alpha.shape[0]):
        val = 1.0 if data_alpha[ii] > thresh else 0.0
        alpha_probs.append(val)
    alpha_probs = np.array(alpha_probs)
    alpha_probs = alpha_probs / np.sum(alpha_probs)

    iterlist = tqdm(range(num_epochs)) if display else range(num_epochs)
    for epoch in iterlist:
        if shuffle:
            np.random.shuffle(order)
            data_x = data_x[order]
            data_y = data_y[order]
            data_alpha = data_alpha[order]
            data_theta = data_theta[order]
            dual_probs = dual_probs[order]
            index = index[order]

        idx1, idx2 = [], []
        didx1, didx2 = [], []
        dtmul1, dtmul2 = [], []
        # For feeding tne Homophily inputs
        for idx in range(data_size):
            # try:
            if np.sum(data_x[idx]) != 0.0:
                p = data_x[idx] / np.sum(data_x[idx])
                samples = np.random.choice(sample_idx, size = 2, p = p)
                idx1.append(samples[0])
                idx2.append(samples[1])

            # except ZeroDivisionError:
            else:
                print 'Exception encountered'
                idx1.append(order[idx])
                idx2.append(order[idx])
                pass

            # try:
            if np.sum(dual_probs[idx]) != 0.0:
                p = dual_probs[idx] / np.sum(dual_probs[idx])
                samples = np.random.choice(sample_idx, size = 2, p = p)

                # samples = np.random.randint(low=0, high=data_size, size=2)
                # samples = np.random.choice(sample_idx, size = 2, p = alpha_probs)
                didx1.append(samples[0])
                didx2.append(samples[1])

                multi1 = data_alpha[idx] * in_alpha[samples[0]] * np.dot(data_theta[idx], in_theta[samples[0]])
                multi2 = data_alpha[idx] * in_alpha[samples[1]] * np.dot(data_theta[idx], in_theta[samples[1]])
                dtmul1.append(multi1)
                dtmul2.append(multi2)

            # except ZeroDivisionError:
            else:
                didx1.append(order[idx])
                didx2.append(order[idx])
                dtmul1.append(0.0)
                dtmul2.append(0.0)

        for batch_num in range(num_batches):
            feed_dict = {}
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # feed_dict["struc_input"] = data_x[start_index:end_index]
            feed_dict["struc_input"] = data_y[start_index:end_index]
            feed_dict["struc_input_neigh1"] = in_y[idx1[start_index : end_index]]
            feed_dict["struc_input_neigh2"] = in_y[idx2[start_index : end_index]]

            # Dual inputs
            feed_dict["input_alpha"] = data_alpha[start_index:end_index]
            feed_dict["dual_input_neigh1"] = in_y[didx1[start_index : end_index]]
            feed_dict["dual_input_neigh2"] = in_y[didx2[start_index : end_index]]
            feed_dict["input_mul1"] = dtmul1[start_index : end_index]
            feed_dict["input_mul2"] = dtmul2[start_index : end_index]

            # for kk in range(batch_size):
            #    print order[kk], feed_dict['input_mul1'][kk], feed_dict['input_mul2'][kk], feed_dict['input_alpha'][kk]

            yield feed_dict, index[start_index: end_index], batch_num == num_batches - 1

def get_activations(sess, model, x_train,
                    margins, radii, centers, penalty3, batch_size):
    batcher = batch_iter_2(x_train, batch_size, 1, False)
    acts = []
    for feed_dict, idx, epoch_end in batcher:
        feed_dict["input_radii_sq"] = np.square(radii[idx])
        feed_dict["input_centers"] = centers[idx]
        feed_dict["input_margin"] = margins[idx]
        feed_dict["input_penalty_coeff"] = penalty3
        act = model.get_act(sess, feed_dict)
        acts.append(act)
    return np.concatenate(acts)

def get_losses(sess, model, x_train, y_train, alphas, thetas, batch_size):
    batcher = batch_iter_2(x_train, y_train, alphas, thetas, batch_size, 1, display=False)
    L1, L2 = [], []
    for feed_dict, idx, epoch_end in batcher:
        l1, l2 = model.get_losses(sess, feed_dict)
        L1.append(l1)
        L2.append(l2)
    L1 = np.concatenate(L1)
    L2 = np.concatenate(L2)

    loss_a = np.mean(L1)
    loss_s = np.mean(L2)
    loss = loss_a + loss_s

    print ('Reconstruction + Homophily Loss: {}'.format(loss_a))
    print ('Dual Loss:                       {}'.format(loss_s))
    print ('Total Loss:                      {}\n'.format(loss))
    # for ii in range(len(L2)):
    #    print(ii, L2[ii])
    return loss

def get_per_point_losses(sess, model, x_train, y_train, batch_size):
    batcher = batch_iter_1(x_train, y_train, batch_size, 1, shuffle = False)
    L1, L2 = [], []
    for feed_dict, epoch_end in batcher:
        l1, l2 = model.get_losses(sess, feed_dict)
        L1.append(l1)
        L2.append(l2)
    L1 = np.concatenate(L1)
    L2 = np.concatenate(L2)

    return L1, L2

def trainer_part1(sess, model, x_train, y_train, num_epochs, batch_size, summ_file, expr_name):
    with sess.as_default():
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

        indexes = np.arange(x_train.shape[0])
        batcher = batch_iter_1(x_train, y_train, batch_size, num_epochs)

        epoch = 0
        print 'Training...'
        for feed_dict, epoch_end in batcher:
            model.train_step(sess, feed_dict, True)
            if epoch_end:
                # path = saver.save(sess, os.path.join(summ_file, 'model.ckpt'))
                # print 'Saved model at', path, 'epoch', epoch
                # print 'Completed epoch', epoch
                # struc_emb = model.get_hidden(sess, y_train)
                # np.savetxt(os.path.join('emb', expr_name + '_part_1_' + str(epoch) + '.emb'), struc_emb)
                # if epoch % 5 == 0:
                #     L1, L2 = get_per_point_losses(sess, model, x_train, y_train, batch_size)
                #     np.savetxt(os.path.join('emb', expr_name + '_L1_' + str(epoch) + '.loss'), L1)
                #     np.savetxt(os.path.join('emb', expr_name + '_L2_' + str(epoch) + '.loss'), L2)
                epoch += 1

        # Only for content branch
        struc_emb = model.get_hidden(sess, y_train)
        path = saver.save(sess, os.path.join(summ_file, 'model.ckpt'))
        print 'Final model saved at', path

        print 'Saving embeddings...', expr_name
        np.savetxt(os.path.join('emb', expr_name + '_part_1' + '.emb'), struc_emb)
        return struc_emb

def trainer_part2(sess, model, x_train, y_train, alphas, thetas,
                    num_epochs, batch_size, summ_file, expr_name):
    with sess.as_default():
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

        indexes = np.arange(x_train.shape[0])
        batcher = batch_iter_2(x_train, y_train, alphas, thetas, batch_size, num_epochs, display=False)

        mode = "all"
        epoch = 0
        print('Training...')
        for feed_dict, idx, epoch_end in batcher:
            model.train_step_dual(sess, feed_dict, mode, False)

            if epoch_end:
                print('Epoch {}.'.format(epoch))
                loss = get_losses(sess, model, x_train, y_train, alphas, thetas, batch_size)
                epoch += 1
        struc_emb = model.get_hidden(sess, y_train)
        path = saver.save(sess, os.path.join(summ_file, 'model.ckpt'))
        return struc_emb
