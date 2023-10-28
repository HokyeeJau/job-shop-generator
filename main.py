# @author: Hokyee Jau
# @date: 2023/10/28

import math
import argparse
from utils import _init_dataset
from conf import JOB_SHOP_CONFIG


def init_dataset(args):
    from copy import deepcopy

    times = 0

    config = deepcopy(JOB_SHOP_CONFIG)
    config['TIMELINE_RANDOM_EMPTY_SIZE'] = args.empty_space_maxima
    config['TIMELINE_LENGTH'] = args.timeline_length
    config['TIMELINE_TIME_MAXIMA'] = config['TIMELINE_LENGTH'] * 0.01 * math.pow(10, args.schedule_digit_num)
    config['JOB_SHOP_DATASET_COUNT'] = args.dataset_num
    config['SCHEDULE_POOL_SIZE'] = args.schedule_pool_size
    config['TIMELINE_POOL_SIZE'] = args.timeline_pool_size
    config['TIMELINE_RANDOM_EMPTY_REPEAT_TIME'] = args.timeline_repeat_time
    config['JOB_SHOP_SCHEDULING_ROOT'] = args.data_root
    config['SCHEDULE_DIGIT_NUM'] = args.schedule_digit_num

    for j in range(config['JOB_SHOP_DATASET_COUNT']):
        _init_dataset(config, times)
        times += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Timeline Generator')
    parser.add_argument('--data_root', type=str, default='jsp/')
    parser.add_argument('--schedule_pool_size', type=int, default=1000)
    parser.add_argument('--timeline_pool_size', type=int, default=1000)
    parser.add_argument('--dataset_num', type=int, default=10)
    parser.add_argument('--timeline_length', type=int, default=8)
    parser.add_argument('--empty_space_maxima', type=int, default=2)
    parser.add_argument('--timeline_repeat_time', type=int, default=2)
    parser.add_argument('--timeline_maxima', type=int, default=8*10)
    parser.add_argument("--schedule_digit_num", type=int, default=3)
    args = parser.parse_args()

    init_dataset(args)