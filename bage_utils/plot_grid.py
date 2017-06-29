from matplotlib import pyplot as plt, gridspec

from bage_utils.plot_util import PlotUtil


class PlotGrid(object):
    def __init__(self, total, n_cols, ax_size=(50, 50)):
        n_rows = total // n_cols + 1
        width = PlotUtil.pixel2inch(ax_size[0] * n_cols)
        height = PlotUtil.pixel2inch(ax_size[1] * n_rows)
        print(width, height)
        self.fig = plt.figure(figsize=(width, height))
        self.gs = gridspec.GridSpec(n_rows, n_cols)
        self.gs.update(left=0.1, right=0.9, bottom=0.5, hspace=1.0)

    def ax(self, nth, title=''):
        _ax = plt.subplot(self.gs[nth])
        _ax.set_title(title)
        return _ax
