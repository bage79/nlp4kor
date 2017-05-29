from collections import defaultdict, Counter  # @UnusedImport


class DimensionUtil(object):
    @staticmethod
    def create_dict(raw_type, dimensions=1):  # @ReservedAssignment
        """ Creates an n-dimension dictionary where the n-th dimension is of type 'type'
        """
        if raw_type != str and raw_type != list and raw_type != dict and raw_type != Counter:
            raise Exception(
                'invalide raw_type. raw_type must be list or dict or Counter')
        if raw_type == list:
            dimensions += 1
        if dimensions <= 1:
            return raw_type()
        return defaultdict(lambda: DimensionUtil.create_dict(raw_type, dimensions - 1))


if __name__ == '__main__':
    m1 = DimensionUtil.create_dict(dict, 2)
    m1['a'] = 1
    print('m1[%s]=%s %s' % ('a', m1['a'], type(m1['a'])))

    m2 = DimensionUtil.create_dict(list, 2)
    m2[1][2] = 2
    print('m2[%s][%s]=%s %s' % ('1', '2', m2[1][2], type(m2[1][2])))

    m4 = DimensionUtil.create_dict(Counter, 4)
    m4['d1']['d2']['d3']['d4'] += 1
    m4['d1']['d2']['d3']['d4'] += 2
    print('m4[%s][%s][%s][%s]=%s %s' % (
        'd1', 'd2', 'd3', 'd4', m4['d1']['d2']['d3']['d4'], type(m4['d1']['d2']['d3']['d4'])))
