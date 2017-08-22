from bage_utils.chunks import Chunks


class ListUtil(object):
    @staticmethod
    def remove_emtpy(li):
        return [line for line in li if len(line) > 0]

    @staticmethod
    def chunks_with_size(li, chunk_size=0, remove_incomplete_item=True):
        """
        list를 각 chunk가 chunk_size크기가 되도록 나눈다.
        모두 chunk_size 크기를 갖고, 마지막에만 가장 작은 배열이 남는다.
        :param li:
        :param chunk_size: 0=나누지 않음.
        :param remove_incomplete_item:
        :return:
        """
        if len(li) < 1 or chunk_size < 1:
            return [li]
        chunk_size = int(chunk_size)

        li2 = []
        for i in range(0, len(li), chunk_size):
            item = li[i: i + chunk_size]
            if remove_incomplete_item and len(item) < chunk_size:
                continue
            li2.append(item)
        return Chunks(li2)

    @staticmethod
    def chunks_with_splits(li, max_split=1, remove_incomplete_item=True):
        """
        list를 chunk의 총 개수가 max_split 개수가 되도록 나눈다.
        모두 똑같은 길이를 갖고, 마지막에만 가장 작은 배열이 남는다.
        :param li:
        :param max_split: 1=나누지 않음.
        :param remove_incomplete_item:
        :return:
        """
        if max_split <= 1:
            return [li]

        min_chunk_size = len(li) // max_split
        return ListUtil.chunks_with_size(li, min_chunk_size, remove_incomplete_item=remove_incomplete_item)

    @staticmethod
    def chunks_banlanced(li, max_split=2):
        """
        list를 max_split 수 대로 균형있게 자른다. (각각 최대한 비슷한 길이가 됨.)
        프로세스에 job을 분배할 때 사용.
        :param li: Split 대상 List
        :param max_split: Split 수
        :return Lists in list
        """
        if max_split < 2:
            return li
        min_chunk_size = len(li) // max_split  # 작은 길이의 청크
        max_chunk_size = min_chunk_size + 1  # 큰 길이의 청크 (작은 길이 + 1)

        if min_chunk_size == 0:
            return [li]
        max_chunk_split = len(li) % max_split
        min_chunk_split = max_split - max_chunk_split

        # print('%s * %s + %s * %s = %s ' % (min_chunk_size, min_chunk_split, max_chunk_size, max_chunk_split, len(li)))
        li2 = []
        li2.extend(list(ListUtil.chunks_with_size(li[:min_chunk_size * min_chunk_split], min_chunk_size)))
        li2.extend(list(ListUtil.chunks_with_size(li[min_chunk_size * min_chunk_split:], max_chunk_size)))
        for a in li2:
            yield a


if __name__ == '__main__':
    li = list(range(1, 51))
    print('li: %s' % li)
    # chunks = ListUtil.chunks(li, 4)

    # print(ListUtil.chunks(li, chunk_size=4))
    # for i, chunks in enumerate(ListUtil.chunks(li, chunk_size=4)):
    for i, chunks in enumerate(ListUtil.chunks_with_splits(li, max_split=6)):
        # for i, chunks in enumerate(ListUtil.chunks_banlanced(li, max_split=13)):
        print('[%04d] %s' % (i, chunks))
