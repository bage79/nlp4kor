import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import numpy as np
import pandas as pd


class PytorchUtil(object):
    random_seed = 7942

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
    pass
