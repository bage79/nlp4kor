import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import pandas as pd
import numpy as np


class PytorchUtil(object):
    random_seed = 7942

    # noinspection PyDefaultArgument
    @staticmethod
    def random_networks(x_dims=3, y_dims=1, total_sample=3,
                        n_hiddens=[10, 50, 100, 1000], n_layers=[2, 3, 4],
                        max_dropout_layers=0, p_dropouts=[0.1, 0.5],
                        max_activation_layers=0, activations=[nn.ReLU, nn.ELU, nn.Tanh, nn.Sigmoid],
                        ):  # TODO: more hyper parameters
        networks = []
        for i in range(total_sample):
            layers = []
            n_layer = np.random.choice(n_layers, 1)[0]  # 레이어 수
            # print('n_layer:', n_layer)
            hidden_in_layers = np.random.choice(n_hiddens, n_layer, replace=True).tolist()
            hidden_in_layers = sorted(hidden_in_layers, reverse=True)
            hidden_in_layers.insert(0, x_dims)
            hidden_in_layers.append(y_dims)
            # print(hidden_in_layers)

            hidden_in_layers_to = hidden_in_layers[1:]
            hidden_in_layers_from = hidden_in_layers[:-1]
            for hidden_from, hidden_to in zip(hidden_in_layers_from, hidden_in_layers_to):
                # print(hidden_from, hidden_to)
                layers.append(nn.Linear(hidden_from, hidden_to))

            if max_dropout_layers > 0 and len(layers) - 1 > 0:
                n_dropout = min(len(layers) - 1, np.random.choice(max_dropout_layers + 1, 1)[0])
                dropout_layer = np.random.choice(len(layers) - 1, n_dropout, replace=False).tolist()
                dropout_layer = sorted(dropout_layer, reverse=True)
                for l in dropout_layer:
                    p = np.random.choice(p_dropouts, 1)[0]
                    layers.insert(l, nn.Dropout(p=p))

            if max_activation_layers > 0 and len(layers) - 1 > 0:
                n_activation = min(len(layers) - 1, np.random.choice(max_activation_layers + 1, 1)[0])
                activation_layer = np.random.choice(len(layers) - 1, n_activation, replace=False).tolist()
                activation_layer = sorted(activation_layer, reverse=True)
                for l in activation_layer:
                    a = np.random.choice(activations, 1)[0]
                    layers.insert(l, a())

            networks.append(nn.Sequential(*layers))
        return networks

    @classmethod
    def init_random_seed(cls, random_seed=None):
        if random_seed is None:
            random_seed = cls.random_seed

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    @staticmethod
    def equal_distribution(df, indexes: list, test_rate=0.1, valid_rate=0.1, shuffle=True):
        df_train, df_valid, df_test = None, None, None
        sums = [i.sum() for i in indexes]
        min_count = min(sums)
        # print('min: %s in %s' % (min_count, sums))

        for index in indexes:
            df_part = df[index]
            df_part = df_part[:min_count]
            if shuffle:
                df_part = df_part.sample(frac=1, random_state=7942)

            n_test = max(1, int(len(df_part) * test_rate))
            n_valid = max(1, int(len(df_part) * valid_rate))
            n_train = len(df_part) - n_test - n_valid
            # print('n_train/n_valid/n_test:', n_train, n_valid, n_test)
            if df_train is None:
                df_train = df_part[:n_train]
            else:
                df_train = df_train.append(df_part[:n_train])

            if df_valid is None:
                df_valid = df_part[n_train:n_train + n_valid]
            else:
                df_valid = df_valid.append(df_part[n_train:n_train + n_valid])

            if df_test is None:
                df_test = df_part[n_train + n_valid:]
            else:
                df_test = df_test.append(df_part[n_train + n_valid:])

        return df_train, df_valid, df_test

    @staticmethod
    def exp_learing_rate_decay(optimizer, epoch, init_lr=0.001, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            # print('LR is set to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return optimizer


if __name__ == '__main__':
    for network in PytorchUtil.random_networks(total_sample=5):
        print(network)
