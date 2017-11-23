import torch
import os
import pandas as pd
import numpy as np
import time


class PytorchUtil(object):
    random_seed = 7942

    @classmethod
    def use_gpu(cls, device_no=0) -> None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_no)

    @classmethod
    def get_gpus(cls) -> int:
        return os.environ.get("CUDA_VISIBLE_DEVICES", [])

    @classmethod
    def init_random_seed(cls, random_seed=None, init_torch=True, init_numpy=True) -> int:
        if random_seed is None:
            random_seed = cls.random_seed

        if init_numpy:
            np.random.seed(random_seed)
        if init_torch:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        return random_seed

    # noinspection PyDefaultArgument
    @classmethod
    def split_dataframe(cls, df, indexes_by_label: list = [], test_rate=0.1, valid_rate=0.1, shuffle=True, full_test=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        df_train, df_valid, df_test = None, None, None
        if len(indexes_by_label) > 0:
            sums = [i.sum() for i in indexes_by_label]
            min_count = min(sums)
            # print('min: %s in %s' % (min_count, sums))
            for index in indexes_by_label:  # same distribution by label
                df_part = df[index]
                df_part = df_part[:min_count]
                if shuffle:
                    df_part = df_part.sample(frac=1, replace=False, random_state=int(time.time()))

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
                    if full_test:
                        df_test = df_part[:]
                    else:
                        df_test = df_part[n_train + n_valid:]
                else:
                    if full_test:
                        df_test = df_test.append(df_part[:])
                    else:
                        df_test = df_test.append(df_part[n_train + n_valid:])

            return df_train, df_valid, df_test
        else:
            if shuffle:
                df = df.sample(frac=1, replace=False, random_state=int(time.time()))

            n_test = int(len(df) * test_rate)
            n_valid = int(len(df) * valid_rate)
            n_train = len(df) - n_test - n_valid

            df_train = df[:n_train]
            df_valid = df[n_train:n_train + n_valid]
            if full_test:
                df_test = df[:]
            else:
                df_test = df[n_train + n_valid:]
            return df_train, df_valid, df_test

    @staticmethod
    def exp_learing_rate_decay(optimizer, epoch, init_lr=0.001, lr_decay_epoch=1) -> torch.optim.Optimizer:
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            # print('LR is set to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return optimizer
