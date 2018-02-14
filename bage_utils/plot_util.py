import math
import os

import matplotlib
import numpy
import pandas
from matplotlib import pyplot, gridspec


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
    def grid_plots(df: pandas.DataFrame, columns=None, second_columns=[], title='', subtitles=[], point_list=[], kind='line', y_min_max=None, second_y_min_max=None, y_label='', plot_columns=1, max_xticks=4, rotate_xtick=45, one_row_height=400, width=2048, title_font_size=50, axhline=True, secondary_y=False, legend=True, grid=True, plot_filepath=None, debug=False):
        matplotlib.rcParams['legend.loc'] = 'upper left'

        if columns is None:
            columns = list(df.columns)

        # print('base_columns:', base_columns)

        plot_rows = math.ceil(len(columns) / plot_columns) + 1  # with title
        figsize_pixel = (width, one_row_height * plot_rows)
        figsize = PlotUtil.pixel2inch(figsize_pixel)
        fig = pyplot.figure(figsize=figsize)  # for len(df)==1000
        gs = gridspec.GridSpec(plot_rows, plot_columns)

        pyplot.subplot(gs[0, :])
        pyplot.axis('off')
        pyplot.text(x=0.5, y=0, s=title, fontsize=title_font_size, horizontalalignment='center', verticalalignment='bottom')
        # fig.suptitle(title, size=title_font_size, horizontalalignment='center', verticalalignment='bottom')

        if len(subtitles) != len(columns):
            subtitles = [str(col) for col in columns]

        for nth, col in enumerate(columns):
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
            if y_label is not None:
                pyplot.ylabel(y_label)

            if kind == 'bar':
                sub_df['pos'] = sub_df[col][sub_df[col] > 0]
                sub_df['neg'] = sub_df[col][sub_df[col] < 0]
                if debug:
                    print(sub_df.head())
                sub_df['pos'].plot.bar(color='r', title=subtitles[nth])
                sub_df['neg'].plot.bar(color='b')
                if len(columns) == len(second_columns):
                    print(second_columns[nth])
                    df[second_columns[nth]].plot.bar(color='g')  # TODO: TEST
                if debug:
                    print('xticks:', len(xticks), xticks)
                pyplot.xticks(xticks, [df.index[idx] for idx in xticks], rotation=0)
            else:
                if len(columns) == len(point_list):
                    x_list, y_list = point_list[nth]
                    sub_df[col].plot.line(title=subtitles[nth], xticks=xticks, markevery=x_list, marker='o', markerfacecolor='red')
                else:
                    sub_df[col].plot.line(title=subtitles[nth], xticks=xticks)

                if len(columns) == len(second_columns):
                    ax2 = df[second_columns[nth]].plot.line(color='g', secondary_y=secondary_y, xticks=xticks)
                    if second_y_min_max is not None:
                        ax2.set_ylim(second_y_min_max)
                pyplot.xticks(xticks, [df.index[idx] for idx in xticks], rotation=0)  # FIXME: doesn't work when secondary_y=True
            if legend:
                pyplot.legend()
            if grid:
                pyplot.grid()

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
