import logging
import os.path
import traceback

import xlrd
import xlwt.Workbook

from bage_utils.dimension_util import DimensionUtil


# import xlutils.copy


class ExcelStyle(object):
    default = xlwt.Workbook(encoding='utf8').get_default_style()

    integer = xlwt.XFStyle()
    integer.font = xlwt.Font()
    integer.num_format_str = '#,##0'
    integer.alignment = xlwt.Alignment()
    integer.alignment.horz = xlwt.Alignment.HORZ_RIGHT
    integer.alignment.vert = xlwt.Alignment.VERT_CENTER

    float = xlwt.XFStyle()
    float.font = xlwt.Font()
    float.num_format_str = '#,##0.00'
    float.alignment = xlwt.Alignment()
    float.alignment.horz = xlwt.Alignment.HORZ_RIGHT
    float.alignment.vert = xlwt.Alignment.VERT_CENTER

    string = xlwt.XFStyle()
    string.font = xlwt.Font()
    string.alignment = xlwt.Alignment()
    string.alignment.horz = xlwt.Alignment.HORZ_CENTER
    string.alignment.vert = xlwt.Alignment.VERT_CENTER


class ExcelUtil(object):
    @staticmethod
    def normalize_sheet_name(sheet_name):
        if (sheet_name is None) or len(sheet_name) == 0:
            return sheet_name
        sheet_name = sheet_name.replace('/', ',')
        return sheet_name


class ExcelWriteSheet(object):
    def __init__(self, writer, sheet, style, log=logging.getLogger()):
        self.sheet = sheet
        self.style = style
        self.writer = writer
        self.log = log

    def __repr__(self):
        return str(self.__dict__)

    def write(self, row, col, val, style=None):
        if style is None:
            self.sheet.write(row, col, val, self.style)
        else:
            self.sheet.write(row, col, val, style)
        return self.writer

    def fill(self, row_list, col_list, val, style=None):
        for row in row_list:
            for col in col_list:
                if style is None:
                    self.sheet.write(row, col, val, None)
                else:
                    self.sheet.write(row, col, val, style)
        return self.writer


class ExcelReadSheet(object):  # aaa

    def __init__(self, reader, sheet, log=logging.getLogger()):
        self.sheet = sheet
        self.reader = reader
        self.log = log

    def __repr__(self):
        return str(self.__dict__)

    def read(self, row, col, default_val=''):
        try:
            value = self.sheet.cell(row, col).value
            if value is None:
                value = default_val
            return value
        except Exception as e:
            self.log.error(traceback.format_exc())
            raise e

    def read_all(self, default_val=''):
        """ read excel file(.xls or .xlsx) and return list of dict """
        try:
            values = DimensionUtil.create_dict(str, 3)
            for row in range(self.sheet.nrows):
                for col in range(self.sheet.ncols):
                    values[row][col] = self.sheet.cell(row, col).value
                    if values[row][col] is None:
                        values[row][col] = default_val
        except Exception as e:
            self.log.error(traceback.format_exc())
            raise e
        return values

    def print_all(self):
        values = self.read_all()
        if len(values) > 0:
            for row in range(self.sheet.nrows):
                for col in range(self.sheet.ncols):
                    print('(%s, %s)=%s' % (row, col, values[row][col]))


class ExcelWriter(object):
    def __init__(self, filepath, style=ExcelStyle.string, encoding='utf8', log=logging.getLogger()):
        self.encoding = encoding
        self.filepath = filepath
        self.ext = os.path.splitext(filepath)[1]
        self.log = log
        self.style = style
        if self.ext.lower() != '.xls' and self.ext.lower() != '.xlsx':
            raise Exception('Available excel file types are .xls and xlsx.')

        try:
            _book = xlrd.open_workbook(self.filepath, formatting_info=True)
            self.book = _book  # xlutils.copy.copy(_book) # FIXME: need test on python3.
        except IOError:  # No such file or directory
            self.book = xlwt.Workbook(encoding='utf8')
            #            self.__add_sheet('Sheet1')
            #            self.save()

    def get_sheet(self, sheet_name):
        sheet_name = ExcelUtil.normalize_sheet_name(sheet_name)
        #        print('sheet_name:', sheet_name
        try:
            book = xlrd.open_workbook(self.filepath)
            sheet_index = book.sheet_by_name(sheet_name).number
            sheet = self.book.get_sheet(sheet_index)
            #            print('sheet_index:', sheet_index
            #            print('sheet_names:', book.sheet_names()
            #            print('sheet:', sheet
            return ExcelWriteSheet(self, sheet, self.style)
        except Exception:
            return self.__add_sheet(sheet_name)
            #            raise Exception('sheet "%s" doest not exists.' % sheet_name)

    def __add_sheet(self, sheet_name):
        sheet_name = ExcelUtil.normalize_sheet_name(sheet_name)
        sheet = self.book.add_sheet(sheet_name, cell_overwrite_ok=True)
        return ExcelWriteSheet(self, sheet, self.style)

    def save(self):
        self.book.save(self.filepath)


class ExcelReader(object):
    def __init__(self, filepath, log=logging.getLogger()):
        self.log = log
        self.filepath = filepath
        self.ext = os.path.splitext(filepath)[1]
        if self.ext.lower() != '.xls' and self.ext.lower() != '.xlsx':
            raise Exception('Available excel file types are .xls and xlsx.')
        self.book = xlrd.open_workbook(self.filepath, formatting_info=True)

    def get_sheet(self, sheet_name):
        sheet_name = ExcelUtil.normalize_sheet_name(sheet_name)
        sheet = self.book.sheet_by_name(sheet_name)
        return ExcelReadSheet(self, sheet)

    def sheet_names(self):
        return self.book.sheet_names()


if __name__ == '__main__':
    out_file_path = 'output/test.xls'
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    ExcelWriter(out_file_path).get_sheet('Sheet1').write(0, 0, 999999).save()
    ExcelReader(out_file_path).get_sheet('Sheet1').print_all()
    print(ExcelReader(out_file_path).get_sheet('Sheet1').read(0, 0))
