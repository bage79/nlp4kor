class PlotUtil(object):
    @staticmethod
    def pixel2inch(pixels, dpi=96):
        if isinstance(pixels, tuple) or isinstance(pixels, list):
            return pixels[0] / dpi, pixels[1] / dpi
        elif isinstance(pixels, int) or isinstance(pixels, float):
            return pixels / dpi
