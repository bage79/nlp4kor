import os

import matplotlib
import numpy
import pandas
from matplotlib import pyplot, gridspec
import math


class PlotUtil(object):
    @staticmethod
    def set_font_for_korean(family="NanumGothic"):
        import matplotlib
        matplotlib.rc('font', family=[family, 'UbuntuMono-BI'])

    @staticmethod
    def pixel2inch(pixels, dpi=96):
        if isinstance(pixels, tuple) or isinstance(pixels, list):
            return pixels[0] / dpi, pixels[1] / dpi
        elif isinstance(pixels, int) or isinstance(pixels, float):
            return pixels / dpi

    @staticmethod
    def font_list():
        return matplotlib.font_manager.get_fontconfig_fonts()
        # return matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    @staticmethod
    def grid_plots(df: pandas.DataFrame, title='', subtitles=[], kind='line', y_min_max=None, plot_columns=1, max_xticks=4, rotate_xtick=45, one_row_height=400, width=2048, title_font_size=50, axhline=True, plot_filepath=None, debug=False):
        matplotlib.rcParams['legend.loc'] = 'upper left'

        plot_rows = math.ceil(len(df.columns) / plot_columns) + 1  # with title
        figsize_pixel = (width, one_row_height * plot_rows)
        figsize = PlotUtil.pixel2inch(figsize_pixel)
        fig = pyplot.figure(figsize=figsize)  # for len(df)==1000
        gs = gridspec.GridSpec(plot_rows, plot_columns)

        fig.suptitle(title, size=title_font_size)
        # print(f'figsize: {figsize_pixel}, total_subplot:{total_subplot}, plot_filepath:{plot_filepath}')

        if len(subtitles) < len(df.columns):
            subtitles = [str(col) for col in df.columns]
        for nth, col in enumerate(df.columns):
            if debug:
                print(f'plot {nth} th...')
            ax = pyplot.subplot(gs[nth + plot_columns])
            if y_min_max is not None:
                ax.set_ylim(y_min_max)
            if axhline:
                ax.axhline(y=0.0, color='black', linestyle=':')
            pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=rotate_xtick)

            sub_df: pandas.DataFrame = df[[col]]
            xticks = numpy.linspace(0, len(df), num=max_xticks, endpoint=True).astype(numpy.int32)
            xticks[-1] -= 1
            if kind == 'bar':
                sub_df['pos'] = sub_df[col][sub_df[col] > 0]
                sub_df['neg'] = sub_df[col][sub_df[col] < 0]
                if debug:
                    print(sub_df.head())
                sub_df['pos'].plot.bar(color='r', title=subtitles[nth])
                sub_df['neg'].plot.bar(color='b')
                if debug:
                    print('xticks:', len(xticks), xticks)
                # print([df.index[idx] for idx in xticks])
                pyplot.xticks(xticks, [df.index[idx] for idx in xticks])
            else:
                if debug:
                    print(sub_df.head())
                sub_df[col].plot.line(title=subtitles[nth], xticks=xticks)

        fig.tight_layout()
        if plot_filepath is None:
            pyplot.show()
        else:
            if os.path.exists(plot_filepath):
                os.remove(plot_filepath)
            pyplot.savefig(plot_filepath)


if __name__ == '__main__':
    for f in PlotUtil.font_list():
        print(f)
