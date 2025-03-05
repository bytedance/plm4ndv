# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import os
from extract_parquet import makedirs
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))

pickle_root_path = os.path.join(current_path, 'data/traversed')
save_root_path = os.path.join(current_path, 'data/split')
makedirs(save_root_path)


def split_data_file(processed_files):
    processed_files = sorted(processed_files)
    train_size = int(len(processed_files) * 0.6)
    test_size = int(len(processed_files) * 0.2)
    valid_size = len(processed_files) - train_size - test_size
    train_files = processed_files[:train_size]
    test_files = processed_files[train_size:train_size+test_size] 
    valid_files = processed_files[-valid_size:]
    return train_files, test_files, valid_files

def dedup_content(file_list, existing_set):
    dedup_cnt = 0
    content = []
    for file in file_list:
        file_path = os.path.join(pickle_root_path, file)
        with open(file_path, 'rb') as f:
            file_content = pickle.load(f)
        content_list = file_content['content_list']

        for table_content in content_list:
            for col_content in table_content:
                if col_content not in existing_set:
                    existing_set.add(col_content)
                    content.append(col_content)
                else:
                    dedup_cnt += 1
    print(dedup_cnt)
    print(f'nun of content {len(existing_set)}')
    return existing_set, content

if __name__ == '__main__':
    pkl_files = os.listdir(pickle_root_path)
    pkl_files = [i for i in pkl_files if i.endswith('.pkl')]

    print(f'{len(pkl_files)} files to be processed')
    train_files, test_files, valid_files = split_data_file(pkl_files)

    train_set, train_data = dedup_content(train_files, set())
    test_set, test_data = dedup_content(test_files, train_set)
    val_set, val_data = dedup_content(valid_files, test_set)

    with open(os.path.join(save_root_path, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(save_root_path, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(save_root_path, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    print('All files are saved!')