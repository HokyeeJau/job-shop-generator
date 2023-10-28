# @author: Hokyee Jau
# @date: 2023/10/26


import os
import json
import math
import random
from pathlib import Path

from typing import List, Dict, Union

import numpy as np
from sklearn.preprocessing import LabelBinarizer


def random_process_time_generator(x):
    t = 0.0
    for i in range(x):
        t += (0.01 * math.pow(10, x)) * random_10_generator(x)
    return t


random_10_generator = lambda x: random.randint(0, 9)


def generate_random_schedule_pool(config) -> List[float]:
    timeline = []
    for i in range(config['SCHEDULE_POOL_SIZE']):
        y = random_process_time_generator(config['SCHEDULE_DIGIT_NUM'])
        if y < 0.009:
            y = 0.01
        timeline.append(y)

    return timeline


def generate_random_timeline(schedule_pool: List[float], config) -> List[float]:
    timeline = random.sample(schedule_pool, config['TIMELINE_LENGTH'] - 1)
    last_schedule = config['TIMELINE_TIME_MAXIMA'] - sum(timeline)
    timeline.append(last_schedule)
    return timeline


def generate_random_timeline_pool(schedule_pool: List[float], config) -> List[List[float]]:
    return [generate_random_timeline(schedule_pool, config) for _ in range(config['TIMELINE_POOL_SIZE'])]


def full_schedules_n_label_last_schedules(
        timeline: List[float],
        schedule_pool: np.ndarray,
        config
) -> Dict[str, List[np.ndarray]]:

    timelines: List[np.ndarray] = [np.array(timeline)]

    label = np.zeros([schedule_pool.shape[0], config['TIMELINE_LENGTH']])
    label[:, -1] = 1

    labels: List[np.ndarray] = [label]
    return dict(timelines=timelines, labels=labels)


def random_delete_schedules_n_label_possible_schedules(
        timeline: List[float],
        schedule_pool: np.ndarray,
        config,
        label_binarizer,
) -> Dict[str, List[np.ndarray]]:

    timelines: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # Repeat the random empty operation on one timeline
    for _ in range(config['TIMELINE_RANDOM_EMPTY_REPEAT_TIME']):

        # Randomly sample n indexes and empty their associative elements in timeline
        empty_indexes = random.sample(
            list(range(0, config['TIMELINE_LENGTH'] - 1)),
            config['TIMELINE_RANDOM_EMPTY_SIZE']
        )

        timeline_array = np.array(timeline)
        empty_one_hot = np.array(label_binarizer.transform(empty_indexes)).sum(axis=0)

        # get raw labels for all empty places
        # [schedule_pool_size, timeline_length]
        _labels = np.zeros([schedule_pool.shape[0], config['TIMELINE_LENGTH']])

        for empty_index in empty_indexes:
            empty_value = timeline[empty_index]

            _labels[:, empty_index] = np.array(empty_value >= schedule_pool, dtype=np.float64)

        # update the priority inside labels
        for empty_index in empty_indexes:

            # the labels in [1, length-1] columns should satisfy the following conditions:
            # - the previous columns are not available to fill the current value
            # - the current column is available to fill the current value
            # - (fill the current value) = (current value is larger or equal to schedules in the pool)
            if empty_index != 0:
                _labels[:, empty_index] = (
                            (1 - np.array(np.sum(_labels[:, :empty_index], axis=1) > 0, dtype=np.float64))
                            * _labels[:, empty_index]
                )

        _labels[:, -1] = 1 - np.array(np.sum(_labels[:, :-1], axis=1) > 0, dtype=np.float64)

        # finally, we get a specifically designed timeline with empty spaces and the labels
        empty_timeline_array = timeline_array * (1 - empty_one_hot)

        timelines.append(empty_timeline_array)
        labels.append(_labels)

    return dict(timelines=timelines, labels=labels)


def rearrange_timelines_labels_into_timeline2schedule2labels(
        original_timeline: List[float],
        timelines: List[np.ndarray],
        labels: List[np.ndarray],
        schedule_pool: np.ndarray) -> List[Dict[str, Union[float, List[float]]]]:

    dataset: List[Dict[str, Union[float, List[float]]]] = []

    for timeline, labels in zip(timelines, labels):
        for target_id in range(schedule_pool.shape[0]):
            _data = dict(
                original=original_timeline,
                timeline=timeline.tolist(),
                target=float(schedule_pool[target_id]),
                label=labels[target_id, :].tolist()
            )
            dataset.append(_data)

    return dataset


def ensure_dataset_dir(config, dir_id) -> str:

    jsp_dir = os.path.join(
        config['JOB_SHOP_SCHEDULING_ROOT'],
        config['JOB_SHOP_DATASET_DIR_FORMAT']
    ).format(
        TIMELINE_LENGTH=str(config['TIMELINE_LENGTH']),
        TIMELINE_TIME_MAXIMA=str(config['TIMELINE_TIME_MAXIMA']),
        TIMELINE_RANDOM_EMPTY_SIZE=config['TIMELINE_RANDOM_EMPTY_SIZE'],
        DIR_IDX=str(dir_id)
    )
    if not Path(jsp_dir).is_dir():
        os.makedirs(jsp_dir)

    return jsp_dir


def get_dataset_path(jsp_dir: str, timeline_idx: int, config) -> str:
    jsp_ds_path = os.path.join(
        jsp_dir,
        config['JOB_SHOP_DATASET_PATH_FORMAT'].format(
            TIMELINE_IDX=str(timeline_idx)
        )
    )

    return jsp_ds_path


def save_timelines_to_json(path: str, dataset: List[Dict[str, Union[float, List[float]]]]):
    with open(path, 'w') as f:
        json.dump(dataset, f)


def _init_dataset(config, dir_id):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(config['TIMELINE_LENGTH']))
    empty_length = config['TIMELINE_RANDOM_EMPTY_SIZE'] + 1

    print(f"{dir_id} Start: timeline_length: {config['TIMELINE_LENGTH']}, empty_size: {config['TIMELINE_RANDOM_EMPTY_SIZE']}")

    # generate schedule pool and timeline pool
    schedule_pool = generate_random_schedule_pool(config)
    timeline_pool = generate_random_timeline_pool(schedule_pool, config)
    schedule_pool = np.array(schedule_pool)

    for empty_length in range(0, empty_length):
        config['TIMELINE_RANDOM_EMPTY_SIZE'] = empty_length
        # Create dataset dir
        jsp_dir = ensure_dataset_dir(config, int(dir_id))

        # iterate timelines to generate incomplete timelines and save them to json files
        for tid in range(len(timeline_pool)):
            timeline = timeline_pool[tid]

            if empty_length != 0:
                _data = random_delete_schedules_n_label_possible_schedules(
                    timeline=timeline,
                    schedule_pool=np.array(schedule_pool),
                    config=config,
                    label_binarizer=label_binarizer
                )
            else:
                _data = full_schedules_n_label_last_schedules(
                    timeline=timeline,
                    schedule_pool=schedule_pool,
                    config=config
                )

            _data = rearrange_timelines_labels_into_timeline2schedule2labels(
                original_timeline=timeline,
                schedule_pool=schedule_pool,
                **_data
            )

            _data_path = get_dataset_path(jsp_dir=jsp_dir, timeline_idx=tid, config=config)
            save_timelines_to_json(path=_data_path, dataset=_data)