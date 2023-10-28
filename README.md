# Machine-agnostic Fixed-length Job-Shop Data Generator

To deal with the limited datasets related to Job-Shop Scheduling Problem, 
a machine-agnostic job-shop data generator is developed. 

This data generator considers several conditions as follows:
- length of timeline
- number of empty blocks 
- length of job-shop
- times of repetition of timelines



## Arguments of Generator
- `dataset_num`: the number of datasets, each of which is generated based on independently sampled job-shops.
- `data_root`: the root directory for saving generated data.
- `timeline_length`: the number of job-shops in each timeline.
- `schedule_pool_size`: the size of job-shop pool where sample `timeline_length` job-shops.
- `timeline_pool_size`: the size of timeline pool.
- `empty_space_maxima`: the maximal number of empty spaces in each timeline. 
- `timeline_repeat_time`: the times of repetition of timelines. 
- `timeline_maxima`: the total time span of each timeline.
- `schedule_digit_num`: the number of digit of each job-shop. Each job-shop is accurate to two decimal places.

### Pseudo code of Workflow
```python
data_set = list()

for dataset_idx in range(dataset_num):
    for empty_length in range(empty_space_maxima):
        schedule_pool = generate_schedule_pool(schedule_pool_size, schedule_digit_num)
        timeline_pool = generate_timeline_pool(schedule_pool, timeline_length, timeline_maxima)
        
        for timeline in timeline_pool:
            for rep_idx in range(timeline_repeat_time):
                
                empty_indexes = sample_index_from_timeline(timeline, empty_length)
                empty_label = generate_one_hot_label_from(empty_indexes, timeline)
                vacated_timeline = vacate_timeline_with_empty_label(empty_label, timeline)
                _data_rows = fill_timeline_with_each_job_shop(vacated_timeline, timeline_pool)
                data_set += _data_rows
```


## Requirements
- `scikit-learn`
- `numpy`


## Usage
```bash 
python main.py --timeline_length 10 --empty_space_maxima 1 
```

