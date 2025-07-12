"""
在QKNOB数据集上对HA算法进行测评，主要参数是几种不同的初始映射策略。
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import product


if __name__ == '__main__':
    with ProcessPoolExecutor(12) as executor:
        pass
