import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tabulate import tabulate

pd.set_option('display.max_columns', None)


class PandasUtil(object):
    """
    - Pandas Util
    """

    def __init__(self, mysql_info: dict):
        self.mysql_engine = create_engine(
            "mysql+mysqldb://{user}:{passwd}@{host}/{db}?charset=utf8".format(**mysql_info),
            encoding='utf8', echo=False)

    def read_sql(self, sql: str, index_col: str = 'date') -> pd.DataFrame:
        df = pd.read_sql(sql=sql, con=self.mysql_engine.raw_connection(), index_col=index_col)
        return df

    def to_sql(self, df: pd.DataFrame, table_name: str, index_label: str = None, dtype: dict = None) -> None:
        if dtype is None:
            dtype = {}

        if index_label:
            df.to_sql(con=self.mysql_engine, name=table_name, if_exists='append', chunksize=10000, dtype=dtype,
                      index=True, index_label=index_label)
        else:
            df.to_sql(con=self.mysql_engine, name=table_name, if_exists='append', chunksize=10000, dtype=dtype,
                      index=False)

    @staticmethod
    def merge_all(df_list: list) -> pd.DataFrame:
        if len(df_list) == 0:
            return pd.DataFrame()
        elif len(df_list) == 1:
            return df_list[0]
        else:
            df = df_list.pop(0)
            for df2 in df_list:
                df = PandasUtil.merge(df, df2)
            return df

    @staticmethod
    def merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        df2_columns = df2.columns.difference(df1.columns)
        return pd.merge(df1, df2[df2_columns], left_index=True, right_index=True, how='outer')

    @staticmethod
    def create_dataframe(rows: list, indexes: list, column_names: list, column_prefix: str = '',
                         extra_column: tuple = None) -> pd.DataFrame:
        if len(column_prefix) > 0:
            _column_names = list(column_names)
            column_names = []
            for c in _column_names:
                column_names.append('%s%s' % (column_prefix, c))

        if len(rows) > 0 and extra_column:
            key, value = extra_column[0], extra_column[1]
            column_names.insert(0, key)
            df = pd.DataFrame(rows, index=indexes, columns=column_names)
            df[key] = pd.Series([value] * len(rows), index=df.index).astype(str)
        else:
            df = pd.DataFrame(rows, index=indexes, columns=column_names)

        df = df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')  # remove duplicated index
        return df

    @staticmethod
    def change_colun_types(df: pd.DataFrame, column_names: list, column_types: list) -> pd.DataFrame:
        for column, _type in zip(column_names, column_types):
            try:
                df[column] = df[column].astype(_type)
            except:
                raise Exception('column={column}, type={_type}, value={value}'.format(column=column, _type=_type,
                                                                                      value=df.iloc[-1][column]))
        return df

    @staticmethod
    def table(df: pd.DataFrame) -> str:
        return tabulate(df, headers='keys', tablefmt='grid', floatfmt=".2f")

    @staticmethod
    def to_json(df: pd.DataFrame) -> str:
        return df.to_json(orient='split')

    @staticmethod
    def read_json(json_str) -> pd.DataFrame:
        """
        JOSN 문자열을 읽어서 DataFrame으로 반환한다.
        Series 인 경우에도 DataFrame으로 변환하여 반환한다.
        :param json_str: 
        :return: pandas.DataFrame or pandas.Series 
        """
        try:
            df = pd.read_json(json_str, orient='split', typ='frame', dtype=False)  # read json as DataFrame
            return df
        except ValueError:  # df_json is not DataFrame
            try:
                s = pd.read_json(json_str, orient='split', typ='series', dtype=False)  # read json as Series
                return s.to_frame(s.name)  # convert series to dataframe
            except:
                return None


if __name__ == '__main__':
    # di = {'data': ['005930'], 'index': ['20170324'], 'columns': ['code']}
    # df = pd.read_json(json.dumps(di), orient='split', dtype=False)
    # # print(df)
    # df_empty = pd.DataFrame({'name': []}, index=[], columns=['name'])
    # # print(df_empty)
    # print(PandasUtil.merge(df, df_empty))
    df1 = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': list('abdabd')},
                       index=[10, 20, 30, 40, 50, 70])
    df2 = pd.DataFrame({'xx': [1, 2, 3, 4, 5, 6], 'yy': list('abdabd')},
                       index=[10, 20, 30, 40, 50, 60])
    df3 = pd.DataFrame({'xxx': [1, 2, 3, 4, 5, 6], 'yyy': list('abdabd')},
                       index=[10, 20, 30, 40, 50, 60])
    df_list = [df1, df1, df2, df3, df1, df3]
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    s_json = PandasUtil.to_json(s)
    s_ = PandasUtil.read_json(s_json)
    print(s_)
    # df_all = PandasUtil.merge_all(df_list)
    # print(PandasUtil.table(df_all))
    # print(df1.iloc[-1]['y'])
    # print(df1[df1['x'] == 1].index[0])
    # df = pd.DataFrame(index=df1.index)
    # for _df in [df1, df2, df3]:
    #     df = df.merge(_df, left_index=True, right_index=True)
    # df_json = PandasUtil.to_json(df1)
    # print(type(df_json))
    # df = PandasUtil.read_json(df_json)
    # print(PandasUtil.table(df))
    #
    # for idx in df.index: # by index
    #     print(type(df.loc[idx]), df.loc[idx])
    # for col in df:  # by columns
    #     print(type(col), df[col])
