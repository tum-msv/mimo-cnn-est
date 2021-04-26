import tensorflow as tf
import os
import sys
from os.path import dirname, realpath, sep, pardir
from joblib import Parallel, delayed
import multiprocessing as mp
from datetime import datetime
from estimators import FastMMSE_mimo, StructuredMMSE_mimo, Genie_mimo, DiscreteMMSE_mimo, \
    ML_mimo, Transforms, GenieOMP_mimo
from training_CNN_mimo import training_worker_pilots, eval_worker_pilots
from modules.Callbacks.MultitrainingCallback import master_progress_bar_process
from pandas import DataFrame
from SCM3GPP.SCMMulti_MIMO import SCMMulti
import csv
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"

if __name__ == '__main__':
    sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)  # add parent directory to path
    if sys.platform == 'darwin':  # set start method for OS X systems
        mp.set_start_method('spawn')
    date_time_now = datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs
    file_name = os.path.splitext(os.path.split(os.path.basename(__file__))[1])[0]

    save_to_file = True
    save_parameters = False
    save_training_error = False

    n_epochs = 2
    n_steps_per_epoch = 40
    n_learning_batche_size = 20

    n_eval_batches = 2
    n_eval_batche_size = 20

    n_path = 3
    snrs = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    # snrs = [5.0]

    n_coherences = 1
    n_antennas_MS = 2
    n_antennas_BS = 64
    n_pilots = 2

    use_other_estimators = False
    use_genie_estimators = False
    use_cnn_estimators = True
    plot_axis = "snr"

    transforms = 'fft'
    activations = ['relu','softmax']
    #activations = ['softmax']

    #n_processes = 1
    n_processes = int(mp.cpu_count() / 2)  # int(mp.cpu_count() / 2 - 1)
    print('Uses ' + str(n_processes) + ' processes')

    path_sigma_MS = 35
    channel = SCMMulti(path_sigma_BS=2.0, path_sigma_MS=path_sigma_MS, n_path=n_path)

    pbar_send, pbar_rec = mp.Pipe()

    # Prepare queue of estimators to rates
    estimators = list()
    cnn_estimators = list()

    print('Init all estimators')
    name_index = 0
    if use_genie_estimators:
        estimators.append(GenieOMP_mimo.GenieOMP(n_pilots, n_antennas_BS, n_antennas_MS, use_rate=False))
    for snr in list(snrs):
        print('for snr = ' + str(snr))
        if use_genie_estimators:
            estimators.append(Genie_mimo.Genie(snr))
        print('  for antennas_BS = ' + str(n_antennas_BS))
        print('  for antennas_MS = ' + str(n_antennas_MS))
        # Conditionally normal estimators
        if use_other_estimators:
            estimators.append(ML_mimo.ML(snr, transform=Transforms.fft))
            estimators.append(FastMMSE_mimo.FastMMSE(channel, snr, n_antennas_BS, n_antennas_MS, n_pilots))
            #estimators.append(
                #StructuredMMSE_mimo.StructuredMMSE(channel, snr, n_antennas_BS, n_antennas_MS, transform=Transforms.fft,
                #                                     n_samples=16 * n_antennas_BS,  n_pilots=n_pilots))
            estimators.append(
                DiscreteMMSE_mimo.DiscreteMMSE(channel, snr, n_antennas_BS, n_antennas_MS,
                                               n_samples=16 * n_antennas_BS, n_pilots=n_pilots))
        if use_cnn_estimators:
            for act in activations:
                weights = None
                est_config = dict(transform=transforms,
                                  activation=act,
                                  shape=(n_coherences, int(n_pilots * n_antennas_BS)),
                                  epochs=n_epochs,
                                  steps_per_epoch=n_steps_per_epoch,
                                  name='cnn_' + '_'.join([transforms, act]) + '_' + str(name_index))
                cnn_estimators.append([channel, snr, est_config, n_learning_batche_size,
                                       n_pilots, n_antennas_BS, n_antennas_MS, pbar_send, weights])
                name_index += 1

    print()
    if use_cnn_estimators:
        init_dict = dict()
        loss = dict()
        val_loss = dict()

        # Initialize single process for overall progressbar
        pro_bar = mp.Process(
            target=master_progress_bar_process, args=(len(cnn_estimators), 'Train CNN estimators', pbar_rec)
        )
        pro_bar.start()

        test = Parallel(n_jobs=n_processes)(
            delayed(training_worker_pilots)(*args) for args in cnn_estimators)

        # Deinitialize overall progressbar
        if sys.platform != 'darwin':
            pbar_send.send(None)
            pro_bar.join()

        estimators.extend(test)

    pbar_send, pbar_rec = mp.Pipe()

    # Prepare queue of estimator to rates
    ts = list()
    for est_iter, est_ in enumerate(estimators):
        if isinstance(est_, tuple):
            est_ = list(est_)
            est_.insert(3, n_pilots)
            est_.insert(4, n_antennas_BS)
            est_.insert(5, n_antennas_MS)
            est_ = tuple(est_)
            estimators[est_iter] = est_
    for _ in range(n_eval_batches):
        for snr in snrs:
            ts.append(
                [channel, snr, estimators, n_eval_batche_size, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots,
                 pbar_send, plot_axis])

    # Initialize single process for overall progressbar
    pro_bar = mp.Process(target=master_progress_bar_process, args=(len(ts), 'Eval all estimators', pbar_rec))
    pro_bar.start()

    results_org = Parallel(n_jobs=n_processes)(
        delayed(eval_worker_pilots)(*args) for args in ts)

    # Deinitialize overall progressbar
    if sys.platform != 'darwin':
        pbar_send.send(None)
        pro_bar.join()
    mses = dict()

    for res in results_org:
        for r in res:
            if r['id'] in mses.keys():
                mses[r['id']] = mses[r['id']] + r['mse'] / n_eval_batches
            else:
                mses[r['id']] = r['mse'] / n_eval_batches

    results = DataFrame()
    for key, mse in mses.items():
        results.loc[key[1], key[0]] = mse
    results = results.rename_axis(index=None, columns=['SNR'])

    name_file = './results/' + file_name + '/' + date_time + '_' + str(n_path) + 'paths_' + \
                str(n_antennas_BS) + 'BS_antennas_' + str(n_antennas_MS) + 'MS_antennas_' + str(path_sigma_MS) + \
                'AS_' + str(n_epochs) + 'epochs_' + str(n_eval_batches) + 'ebatch_' + \
                str(n_learning_batche_size) + 'lbatchsize_' + str(n_pilots) + 'pilots'

    file_path = name_file + '.csv'
    if save_to_file:
        is_file = os.path.isfile(file_path)
        with open(file_path, 'a+',
                  newline='') as f:  # fix additional empty rows in output csv file in windows by adding: newline=''
            results.sort_index(inplace=True)
            results.to_csv(f)

    if save_parameters and use_cnn_estimators:
        file_path = name_file + '_params.csv'
        for iter, snr in enumerate(snrs):
            result = test[iter][5]
            parameters = DataFrame()
            for key, para in result.items():
                parameters.loc[key, activations[0]] = para
            is_file = os.path.isfile(file_path)
            with open(file_path, 'a+',
                      newline='') as f:  # fix additional empty rows in output csv file in windows by adding: newline=''
                # results.sort_index(inplace=True)
                parameters.to_csv(f, header=not is_file)

    if save_training_error and use_cnn_estimators:  # only meaningful fo a specific snr value
        file_path = name_file + '_trainError.csv'
        with open(file_path, 'w') as myfile:
            losses = []
            for i in range(len(activations)):
                dict_ = test[i][4]
                dict_ = dict_.get('loss')
                temp = []
                for it in np.arange(0, len(dict_), n_steps_per_epoch):
                    temp.append(np.mean(dict_[it:it + n_steps_per_epoch]))
                temp.insert(0, activations[i])
                losses.append(temp)
            if len(activations) == 2:
                loss = []
                for i in range(len(losses[0])):
                    if i == 0:
                        loss.append(['iter', losses[0][i], losses[1][i]])
                    else:
                        loss.append([i, losses[0][i], losses[1][i]])
            elif len(activations) == 1:
                loss = []
                for i in range(len(losses[0])):
                    if i == 0:
                        loss.append(['iter', losses[0][i]])
                    else:
                        loss.append([i, losses[0][i]])
                # losses = zip(zip(range(1,len(losses[0])+1),losses[0],losses[1]))
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(loss)

        file_path = name_file + '_valError.csv'
        with open(file_path, 'w') as myfile:
            losses = []
            for i in range(len(activations)):
                dict_ = test[i][4]
                dict_ = dict_.get('val_loss')
                temp = []
                for it in np.arange(0, len(dict_), n_steps_per_epoch):
                    temp.append(np.mean(dict_[it:it + n_steps_per_epoch]))
                temp.insert(0, activations[i])
                losses.append(temp)
            if len(activations) == 2:
                loss = []
                for i in range(len(losses[0])):
                    if i == 0:
                        loss.append(['iter', losses[0][i], losses[1][i]])
                    else:
                        loss.append([i, losses[0][i], losses[1][i]])
            elif len(activations) == 1:
                loss = []
                for i in range(len(losses[0])):
                    if i == 0:
                        loss.append(['iter', losses[0][i]])
                    else:
                        loss.append([i, losses[0][i]])
                # losses = zip(zip(range(1,len(losses[0])+1),losses[0],losses[1]))
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(loss)

    print(results)

    if sys.platform == 'darwin':
        pbar_send.send(None)
        pro_bar.join()
