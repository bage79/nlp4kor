from enum import Enum as E


# noinspection PyClassHasNoInit
class EnumUtil(E):
    def __int__(self):
        return self.value

    def __str__(self):
        return self.name


if __name__ == '__main__':
    # noinspection PyClassHasNoInit
    class Market(EnumUtil):
        장내 = 1
        코스닥 = 10
        ELW = 3
        ETF = 8
        KONEX = 50
        뮤추얼펀드 = 4
        신주인수권 = 5
        리츠 = 6
        하이얼펀드 = 9
        KOTC = 30


    # noinspection PyClassHasNoInit
    class Quantity(EnumUtil):
        수량 = 0
        금액_백만원 = 1


    # print(Market.장내)
    # print(Market.장내.name)
    # print(Market.장내.value)
    # print(int(Market.장내))
    print(Market(1))
    print(Market(1).name)
    print(Market(1).value)
