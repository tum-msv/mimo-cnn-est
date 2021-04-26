from sys import stderr

import numpy as np
import six
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import multiprocessing as mp
from random import shuffle


class MultitrainingCallback(Callback):
    def __init__(self,
                 pipe_connection,
                 show_hier_training=False,
                 show_training=True,
                 show_epoch=False,
                 leave_hier_training=False,
                 leave_training=False,
                 leave_epoch=False,
                 hier_training_description="Hierarchical training",
                 training_description="Training",
                 epoch_description_initial="Epoch: {epoch}",
                 epoch_description_update="Epoch: {epoch} - {metrics}",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 output_file=stderr,
                 initial=0,
                 snrs = None,
                 est = None,
                 list_of_eval_snrs = None,
                 validation_set = None,
                 batch_size = None,
                 snr_range_counter = None,
                 number_epochs = None):
        super(MultitrainingCallback, self).__init__()
        if pipe_connection is not mp.connection:
            ValueError('{} not a valid pipe connection'.format(pipe_connection))
        self.pipe_connection = pipe_connection
        self.process_id = mp.current_process().pid
        self.show_epoch = show_epoch
        self.show_training = show_training
        self.show_hier_training = show_hier_training
        self.training_description = training_description
        self.hier_training_description = hier_training_description
        self.epoch_description_initial = epoch_description_initial
        self.epoch_description_update = epoch_description_update
        self.metric_format = metric_format
        self.separator = separator
        self.leave_epoch = leave_epoch
        self.leave_training = leave_training
        self.leave_hier_training = leave_hier_training
        self.output_file = output_file
        self.tqdm_outer = None
        self.tqdm_inner = None
        self.epoch = None
        self.running_logs = None
        self.inner_count = None
        self.initial = initial
        self.snrs = snrs
        self.est = est
        self.list_of_eval_snrs = list_of_eval_snrs
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.snr_range_counter = snr_range_counter
        self.number_epochs = number_epochs
        self.snr_list_per_epoch = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        desc = self.epoch_description_initial.format(epoch=self.epoch)
        self.mode = 0  # samples
        if 'samples' in self.params:
            self.inner_total = self.params['samples']
        elif 'nb_sample' in self.params:
            self.inner_total = self.params['nb_sample']
        else:
            self.mode = 1  # steps
            self.inner_total = self.params['steps']
        if self.show_epoch:
            self.pipe_connection.send([str(self.process_id) + '_epoch', dict(desc=desc,
                                                                        total=self.inner_total,
                                                                        leave=self.leave_training)])
        self.inner_count = 0
        self.running_logs = {}


    def on_epoch_end(self, epoch, logs={}):
        from modules import Losses
        if self.show_epoch:
            self.pipe_connection.send([str(self.process_id) + '_epoch', None])
        if self.show_training:
            self.pipe_connection.send([str(self.process_id) + '_training', dict(update=1)])

        if self.snr_range_counter is not None:
            if epoch == int(0.8*self.number_epochs):
                self.snr_range_counter += 1
            elif  epoch == int(0.85*self.number_epochs):
                self.snr_range_counter += 1
            elif epoch == int(0.9 * self.number_epochs):
                self.snr_range_counter += 1
            elif epoch == int(0.95 * self.number_epochs):
                self.snr_range_counter += 1


    def on_batch_end(self, batch, logs={}):
        if self.mode == 0:
            update = logs['size']
        else:
            update = 1
        self.inner_count += update

        #if self.inner_count < self.inner_total:
        #    self.append_logs(logs)
        #    metrics = self.format_metrics(self.running_logs)
        #    desc = self.epoch_description_update.format(epoch=self.epoch, metrics=metrics)
        #    if self.show_training:
        #        # self.tqdm_inner.desc = desc
        #        self.pipe_connection.send([str(self.process_id) + '_epoch', dict(update=update)])

    def on_train_begin(self, logs={}):
        if self.show_training:
            epochs = (self.params['epochs'] if 'epochs' in self.params
                      else self.params['nb_epoch'])
            self.pipe_connection.send([str(self.process_id) + '_training', dict(total=epochs,
                                                                           desc=self.training_description,
                                                                           leave=self.leave_training)])

    def on_train_end(self, logs={}):
        if self.show_training:
            self.pipe_connection.send([str(self.process_id) + '_training', None])

        if self.show_hier_training:
            self.pipe_connection.send([str(self.process_id) + '_hier_training', dict(update=1)])

    def on_hier_train_begin(self, steps):
        if self.show_hier_training:
            self.pipe_connection.send([str(self.process_id) + '_hier_training', dict(total=steps,
                                                                                desc=self.hier_training_description,
                                                                                leave=self.leave_hier_training)])

    def on_hier_train_end(self):
        if self.show_hier_training:
            self.pipe_connection.send([str(self.process_id) + '_hier_training', 'top_level'])


    def append_logs(self, logs):
        metrics = self.params['metrics']
        for metric, value in six.iteritems(logs):
            if metric in metrics:
                if metric in self.running_logs:
                    self.running_logs[metric].append(value[()])
                else:
                    self.running_logs[metric] = [value[()]]

    def format_metrics(self, logs):
        metrics = self.params['metrics']
        strings = [self.metric_format.format(name=metric, value=np.mean(logs[metric], axis=None)) for metric in metrics
                   if
                   metric in logs]
        return self.separator.join(strings)


# Overall progressbar
def master_progress_bar_process(total, description, pipe_connection):
    pbar = tqdm(total=total, position=0)
    training_bar_positions = list(reversed(range(1, total + 1)))
    pbar.set_description(description)
    id_dict = dict()
    for te in iter(pipe_connection.recv, None):
        id, value = te
        if id in id_dict:
            if isinstance(value, dict) and 'update' in value.keys():
                id_dict[id].update(value.pop('update'))
            else:
                training_bar_positions.append(-id_dict[id].pos)
                training_bar_positions.sort(reverse=True)
                id_dict[id].close()
                id_dict.pop(id)
                if isinstance(value, str) and 'top_level' == value:
                    pbar.update(1)
        elif value:
            if len(training_bar_positions) == 1:
                training_bar_positions = list(reversed(range(training_bar_positions[0], training_bar_positions[0] + total + 1)))
            id_dict[id] = tqdm(position=training_bar_positions.pop(), **value)
        else:
            pass
