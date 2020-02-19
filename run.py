import numpy as np
import tensorflow as tf
import os
import json
import csv
from tqdm import tqdm
from datetime import datetime
from preprocessRW import computeRep
from collections import defaultdict, Counter
from copy import deepcopy

from model import AutoEncoder
from neural_training import *
from non_neural_training import *

start_time = datetime.now()

def load_config(fname='./config'):
    with open(fname, 'r') as fp:
        config = json.load(fp)
    return config

def read_csv_file_as_numpy(fname):
    with open(fname, 'r') as fp:
        rd = csv.reader(fp)
        ret = []
        for row in tqdm(rd):
            ret.append([float(r) for r in row])
    return np.array(ret)

if __name__ == "__main__":
    # Load Config
    config = load_config()

    # Read Data
    print 'Reading structure from', config['struc_file']
    Adj = read_csv_file_as_numpy(config['struc_file'])
    Adj = computeRep(Adj, 2, 0.3)
    Con = deepcopy(Adj)
    Con[Con > 0] = 1.0

    # Only for content brach
    config['struc_size'] = Con.shape[1]

    # Build Network
    model = AutoEncoder(config)
    model.create_network()
    model.initialize_optimizer(config)

    # Create Session
    session_conf = tf.ConfigProto(allow_soft_placement = True,
                                  log_device_placement = False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config = session_conf)
    summ_file = './log/' + config['experiment_name']
    model.initialize_summary_writer(sess, summ_file)
    sess.run(tf.global_variables_initializer())

    batch_size = config['batch_size'] if config['batch_size'] > 0 else Adj.shape[0]
    # num_epochs = config['num_epochs']

######## Algorithm part 1 ########
    # 1. Pretrain AutoEncoder
    datapoints = trainer_part1(sess, model, Adj, Con, 20, batch_size, # 20
                                        summ_file, config['experiment_name'])
    print('Part 1: Pre-training completed.\n')

######## Algorithm part 2 ########
    solvers.options['show_progress'] = True
    solvers.options['maxiters'] = 100

    thetas = perform_kmeans(datapoints, config['num_clusters'])
    idx = []
    cnt = 0
    for ii in range(len(thetas)):
        idx.append(np.argmax(thetas[ii]))
        if np.prod(thetas[ii]) == 0:
            cnt += 1

    nu = config['nu']
    for tt in range(15):
        alphas, status = update_alphas(datapoints, thetas, nu)
        print('Iter: {}. Part 2a: alphas training completed.\n'.format(tt))

        # thetas_new = update_thetas(datapoints, thetas, alphas, num_epochs = 5)
        thetas_new = update_thetas_2(datapoints, thetas, alphas, nu)
        print('Iter: {}. Part 2b: Theta training completed.\n'.format(tt))

        datapoints = trainer_part2(sess, model, Adj, Con, alphas, thetas,
                                    4, batch_size, summ_file, config['experiment_name'])
        print('Iter: {}. Part 2b: Neural Training completed.\n'.format(tt))
        thetas = thetas_new

    np.savetxt(os.path.join('emb', config['experiment_name'] + '.emb'), datapoints)
    np.savetxt(os.path.join('emb', config['experiment_name'] + '.alp'), alphas)
    np.savetxt(os.path.join('emb', config['experiment_name'] + '.the'), thetas)
    sess.close()
    end_time = datetime.now()
    print('This run of the algorithm took ')
    print((end_time - start_time).seconds)
