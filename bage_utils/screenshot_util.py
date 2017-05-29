from selenium import webdriver

from bage_utils import base_util


class ScreenshotUtil(object):
    """
    - Create web page screenshot by `Selenium`.
    """

    @staticmethod
    def from_url(url, image_path=__file__ + '.png'):
        #    print sys.platform
        #    if sys.platform.startswith('linux'):
        #        from pyvirtualdisplay import Display
        #        display = Display(visible=0, size=(1024, 768))
        #        display.start()
        #    os.environ["DISPLAY"] = "127.0.0.1:0"
        browser = webdriver.Firefox()
        browser.get(url)
        browser.save_screenshot(image_path)
        browser.quit()
        #    if sys.platform.startswith('linux'):
        #        display.stop()


if __name__ == '__main__':
    ScreenshotUtil.from_url('http://www.naver.com/', base_util.real_path('output/screenshot.png'))
