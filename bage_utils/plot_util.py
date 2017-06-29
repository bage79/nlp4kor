class PlotUtil(object):
    @staticmethod
    def pixel2inch(pixels):
        if isinstance(pixels, tuple) or isinstance(pixels, list):
            return pixels[0] / 96, pixels[1] / 96
        elif isinstance(pixels, int) or isinstance(pixels, float):
            return pixels / 96
