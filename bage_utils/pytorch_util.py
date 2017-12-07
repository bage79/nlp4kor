import traceback

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
    def create_random_seed(cls) -> int:
        return int(time.time() * 10e7) % 2 ** 32
    
    @classmethod
    def init_random_seed(cls, random_seed=None, init_torch=True, init_numpy=True) -> int:
        if random_seed is None:
            random_seed = cls.random_seed

        if init_numpy:
            np.random.seed(random_seed)
        if init_torch:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():  # torch.nn.Module.cuda() first
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
        return random_seed

    @classmethod
    def exp_learing_rate_decay(cls, optimizer, epoch, init_lr=0.001, lr_decay_epoch=1) -> torch.optim.Optimizer:
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            # print('LR is set to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return optimizer

    # noinspection PyDefaultArgument
    @classmethod
    def random_datasets(cls, df, indexes_by_label: list = [], test_rate=0.1, valid_rate=0.1, shuffle=True, full_test=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """

        :param df: pandas.DataFrame
        :param indexes_by_label: [index_of_negative, index_of_positive]
        :param test_rate: test data rate
        :param valid_rate: valid data rate
        :param shuffle: shuffle in each label
        :param full_test: test data == full data (전체 데이터를 테스트 데이터로 쓸지 여부
        :return:
        """
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

    @classmethod
    def cross_valid_buckets(cls, length, max_cross_validation=10):
        """
        minimize the rest (which will be dropped).

        if length >=10 and max_cross_validation == 10, buckets(return value) will be 5 ~ 10

        :param length: total data length
        :param max_cross_validation: initial bucket size
        :return:
        """
        data_in_bucket = int(length / max_cross_validation) if length % max_cross_validation == 0 else int(length / max_cross_validation) + 1
        return int(length / data_in_bucket)

    # noinspection PyDefaultArgument
    @classmethod
    def cross_valid_datasets(cls, df, max_cross=10, nth_data=None, indexes_by_label: list = [], full_test=False, shuffle_sample=True, change_cross=True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, int):
        """

        :param df: pandas.DataFrame
        :param max_cross: number of buckets
        :param nth_data: nth bucket data, bucket[nth]=test data, bucket[nth+1]=valid data
        :param indexes_by_label: [negative_index, positive_index]
        :param full_test: return all data or one bucket
        :param shuffle_sample: shuffle when sampling is needed. len(positive) != len(negative)
        :param change_cross: change max_cross for fitting to data size
        :return:
        """
        if nth_data is None:
            nth_data = 0

        df_train, df_valid, df_test, n_cross = None, None, None, max_cross
        if indexes_by_label is not None and len(indexes_by_label) > 0:
            sums = [i.sum() for i in indexes_by_label]
            min_count = min(sums)
            # print('min: %s in %s' % (min_count, sums))
            for index in indexes_by_label:  # same distribution by label
                df_part = df[index]
                if len(df_part) > min_count:
                    if shuffle_sample:
                        df_part = df_part.sample(min_count, replace=False, random_state=nth_data)
                    else:
                        df_part = df_part[-min_count:]
                    # print(nth_data, df_part.head())

                _df_train, _df_valid, _df_test, n_cross = cls.cross_valid_datasets(df_part, max_cross=max_cross, nth_data=nth_data, full_test=full_test, change_cross=change_cross)

                if df_train is None:
                    df_train = _df_train
                else:
                    df_train = df_train.append(_df_train)

                if df_valid is None:
                    df_valid = _df_valid
                else:
                    df_valid = df_valid.append(_df_valid)

                if df_test is None:
                    df_test = _df_test
                else:
                    df_test = df_test.append(_df_test)

            return df_train, df_valid, df_test, n_cross
        else:
            if len(df) < 3:
                return None, None, None, max_cross  # can't create dataaset
            elif 3 <= len(df) < max_cross:
                max_cross = 3

            if change_cross:
                n_cross = cls.cross_valid_buckets(len(df), max_cross)
            else:
                n_cross = max_cross
            data_in_bucket = int(len(df) / n_cross)

            # print('len(df):', len(df))
            # print('n_cross:', n_cross)
            # print('data_in_bucket:', data_in_bucket)
            # print('n_cross:', n_cross)

            if nth_data is None:
                nth_data = 0
                # nth_data = int(np.random.choice(n_cross, 1, replace=False)[0])

            df = df[-n_cross * data_in_bucket:]
            test_no = nth_data % n_cross
            valid_no = (nth_data + 1) % n_cross
            for i in range(n_cross):
                df_part = df[i * data_in_bucket: (i + 1) * data_in_bucket].copy()
                if i == test_no:
                    if full_test:
                        df_test = df[:]
                    else:
                        df_test = df_part
                elif i == valid_no:
                    df_valid = df_part
                else:
                    if df_train is None:
                        df_train = df_part
                    else:
                        df_train = df_train.append(df_part)
        return df_train, df_valid, df_test, n_cross

    # noinspection PyDefaultArgument
    @classmethod
    def random_layers(cls, x_dims=3, y_dims=1,
                      n_hiddens=[10, 50, 100, 1000], n_layers=[1, 2, 3, 4], hiddens_descending=True,
                      max_dropout_layers=0, p_dropouts=[0.1, 0.5], dropout_low_layers=True,
                      max_activation_layers=1, activations=[torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh, torch.nn.Sigmoid],
                      batch_normal=False):
        try:
            layers = []
            first_dropout_layer = 1  # exclude data layer(=0)
            first_activation_layer = 1  # exclude data layer(=0)
            n_layer = np.random.choice(n_layers, 1)[0]  # 레이어 수

            hidden_in_layers = np.random.choice(n_hiddens, n_layer, replace=True).tolist()
            if hiddens_descending:
                hidden_in_layers = sorted(hidden_in_layers, reverse=True)
            hidden_in_layers.insert(0, x_dims)
            hidden_in_layers.append(y_dims)

            hidden_in_layers_to = hidden_in_layers[1:]
            hidden_in_layers_from = hidden_in_layers[:-1]
            for hidden_from, hidden_to in zip(hidden_in_layers_from, hidden_in_layers_to):
                layers.append(torch.nn.Linear(hidden_from, hidden_to))

            if max_dropout_layers > 0 and len(layers) - 1 > 0:
                n_dropout = min(len(layers) - 1, np.random.choice(max_dropout_layers + 1, 1)[0])
                if dropout_low_layers:
                    dropout_layer = np.random.choice([0, 1], n_dropout, replace=False).tolist()
                else:
                    dropout_layer = np.random.choice([i for i in range(first_dropout_layer, len(layers))], n_dropout, replace=False).tolist()
                dropout_layer = sorted(dropout_layer, reverse=True)
                for l in dropout_layer:
                    p = np.random.choice(p_dropouts, 1)[0]
                    layers.insert(l, torch.nn.Dropout(p=p))

            if max_activation_layers > 0 and len(layers) - 1 > 0:
                n_activation = min(len(layers) - 1, np.random.choice(max_activation_layers + 1, 1)[0])
                activation_layer = np.random.choice([i for i in range(first_activation_layer, len(layers))], n_activation, replace=False).tolist()
                activation_layer = sorted(activation_layer, reverse=True)
                for l in activation_layer:
                    a = np.random.choice(activations, 1)[0]
                    layers.insert(l, a())

            if batch_normal:
                n_batch_normal = np.random.choice(2, 1)[0]  # 0 or 1
                if n_batch_normal == 1:
                    layers.insert(0, torch.nn.BatchNorm1d(x_dims))

            return layers
        except:
            traceback.print_exc()
            return None


if __name__ == '__main__':
    for n in [2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 31, 41, 51, 61, 71, 81, 100, 101, 111, ]:
        print(n, PytorchUtil.cross_valid_buckets(n))
    # df = pd.DataFrame(data=np.arange(11), columns=['a'])
    # print(len(df), df)
    # negative_index = df['a'] <= 10
    # positive_index = df['a'] > 10
    # # print(negative_index)
    # max_cross_validation = 10
    # pick_no = 0
    #
    # # n_cross = PytorchUtil.cross_valid_buckets(len(df), max_cross_validation=max_cross_validation)
    # # print('n_cross:', n_cross)
    #
    # # for i in range(3):
    # for pick_no in range(max_cross_validation):
    #     # df_train, df_valid, df_test, _n_cross = PytorchUtil.cross_valid_datasets(df, indexes_by_label=[negative_index, positive_index], n_cross=n_cross, nth_data=pick_no)
    #     df_train, df_valid, df_test, _n_cross = PytorchUtil.cross_valid_datasets(df, max_cross=max_cross_validation, nth_data=pick_no, full_test=True)
    #     print()
    #     print('%s -> %s' % (max_cross_validation, _n_cross))
    #     print('df_test:', df_test)
    #     print('df_valid:', df_valid)
    #     print('df_train:', df_train)
    #     if pick_no >= _n_cross - 1:
    #         break
