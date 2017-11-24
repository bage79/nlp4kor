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
    def init_random_seed(cls, random_seed=None, init_torch=True, init_numpy=False) -> int:
        if random_seed is None:
            random_seed = cls.random_seed

        if init_numpy:
            np.random.seed(random_seed)
        if init_torch:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():  # nn.Module.cuda() first
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

    # noinspection PyDefaultArgument
    @classmethod
    def cross_valid_datasets(cls, df, indexes_by_label: list = [], n_cross=10, pick_no=None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """

        :param df: pandas.DataFrame
        :param indexes_by_label: [negative_index, positive_index]
        :param n_cross: number of buckets
        :param pick_no: nth data from splitted
        :return:
        """
        if pick_no is None:
            pick_no = int(np.random.choice(n_cross, 1)[0])

        pick_no = 8

        df_train, df_valid, df_test = None, None, None
        if len(indexes_by_label) > 0:
            sums = [i.sum() for i in indexes_by_label]
            min_count = min(sums)
            # print('min: %s in %s' % (min_count, sums))
            for index in indexes_by_label:  # same distribution by label
                df_part = df[index]
                df_part = df_part[-min_count:]

                _df_train, _df_valid, _df_test = cls.cross_valid_datasets(df_part, n_cross=n_cross, pick_no=pick_no)

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

            return df_train, df_valid, df_test
        else:
            if len(df) < 3:
                return None, None, None  # can't create dataaset
            elif 3 <= len(df) < n_cross:
                n_cross = 3

            data_in_bucket = int(len(df) / n_cross) + 1  # with dummy
            n_cross = int(len(df) / data_in_bucket)

            test_no = pick_no % n_cross
            valid_no = (pick_no + 1) % n_cross
            for i in range(n_cross):
                df_part = df[i * data_in_bucket: (i + 1) * data_in_bucket].copy()
                if i == test_no:
                    df_test = df_part
                elif i == valid_no:
                    df_valid = df_part
                else:
                    if df_train is None:
                        df_train = df_part
                    else:
                        df_train = df_train.append(df_part)
        return df_train, df_valid, df_test


if __name__ == '__main__':
    df = pd.DataFrame(data=np.arange(22), columns=['a'])
    negative_index = df['a'] <= 10
    positive_index = df['a'] > 10
    # print(negative_index)
    n_cross = 10
    for pick_no in range(n_cross):
        df_train, df_valid, df_test = PytorchUtil.cross_valid_datasets(df, indexes_by_label=[negative_index, positive_index], n_cross=n_cross, pick_no=pick_no)
        print('df_test:', df_test)
        # print('df_train:', df_train)
        # break
