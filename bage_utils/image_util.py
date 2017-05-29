import logging
import os.path
import traceback

from PIL import Image, ImageFont, ImageDraw, ImageOps  # pip install pillow

from bage_utils import base_util
from bage_utils.dic2object_util import Dic2Object


class ImageUtil(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_format = os.path.splitext(self.img_path)[1]
        self.img = Image.open(img_path)

    def size(self):
        return self.img.size

    def resize(self, size):
        return self.img.resize(size)

    def crop(self, crop_box):
        return self.img.crop(crop_box)

    def save(self, img_path=None):
        if not img_path:
            img_path = self.img_path
        img_ext = os.path.splitext(img_path)[1][1:]
        if img_ext.upper() == 'JPG':
            img_ext = 'JPEG'
        self.img.save(img_path, img_ext)

    def center_pos(self, bg_width, bg_height):
        """
        get center position.
        """
        return (bg_width - self.img.size[0]) / 2, (bg_height - self.img.size[1]) / 2

    def write_text(self, text_position, text_value, fonts_path, fonts_size=11, logger=logging.getLogger(__file__)):
        """
        :param text_position: dictonary of text_positions. e.g. {'name':(0,100), ...}
        :param text_value: dictonary of names. e.g. {'name':'Hyewoong Park', ...}
        :param fonts_path: font file path
        :param fonts_size: font size
        """
        try:
            font = ImageFont.truetype(fonts_path, fonts_size)

            for key, text in text_value.items():
                name_image = Image.new('L', font.getsize(text))
                name_draw = ImageDraw.Draw(name_image)
                name_draw.text((0, 0), text, font=font, fill=255)
                #        text_positions[key] = center_pos(img, nameImage, 0)
                if key in text_position.keys():
                    self.img.paste(ImageOps.colorize(name_image, (255, 255, 255), (0, 0, 0)), text_position[key],
                                   name_image)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise e


if __name__ == '__main__':
    user = Dic2Object({'pk': 123, 'name': '홍길동', 'email': 'kildong.hong@gmail.com'})

    image = ImageUtil(base_util.real_path('input/card_bg_216x156.png'))
    out_path = base_util.real_path('output/card.jpg')

    text_positions = {u'pk': (0, 0), u'name': (0, 30), u'email': (0, 60)}
    font_path = base_util.real_path('input/SeoulNamsanB.ttf')
    font_size = 11
    text_values = {u'pk': str(user.pk), u'name': user.name, u'email': user.email}

    print(image.size())
    if os.path.exists(out_path):
        os.remove(out_path)
    image.write_text(text_positions, text_values, font_path, font_size)
    image.save(out_path)

    # image.resize((600, 600)).save('output/card.jpg')
    # print(image.size()
