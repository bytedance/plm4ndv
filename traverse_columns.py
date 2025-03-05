# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import os
import re
import pickle
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
from extract_parquet import makedirs

current_path = os.path.dirname(os.path.abspath(__file__))

pickle_root_path = os.path.join(current_path, 'data/extracted')
save_root_path = os.path.join(current_path, 'data/traversed')
makedirs(save_root_path)


def filter_types(tab_name, tab_type):
    filtered_name = []
    filtered_type = []
    for i in range(len(tab_name)):
        # ignore composite/null types
        if str(tab_type[i]) != 'dict' and str(tab_type[i]) != 'list' and str(tab_type[i]) != 'null':
            filtered_name.append(tab_name[i])
            filtered_type.append(tab_type[i])
    return filtered_name, filtered_type

def filter_names(col_name_str):
    if len(col_name_str) <= 1:
        return True
    if 'Unnamed' in str(col_name_str):
        return True
    if bool(re.fullmatch(r'[0-9. /-]+', str(col_name_str))):
        return True
    if bool(re.fullmatch(r'^[0-9. ]*e\+[0-9]+$', str(col_name_str))) or bool(re.fullmatch(r'^[0-9. ]*e\-[0-9]+$', str(col_name_str))):
        return True
    if bool(re.search(r'\d', str(col_name_str))):
        return True
    return False

def common_prefix(strs):
    if not strs or len(strs) <= 2:
        return ""
    
    prefix = strs[2]
    
    for s in strs[3:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]  
            if not prefix:
                return ""
    return prefix

def col_level_data_process(extracted_file_name):
    with open(os.path.join(pickle_root_path, extracted_file_name), 'rb') as f:
        extracted_pkl = pickle.load(f)
    table_N_list = extracted_pkl['table_N_list']
    table_D_list = extracted_pkl['table_D_list']
    table_col_name = extracted_pkl['table_col_name']
    table_types = extracted_pkl['table_types']
    sample_list = extracted_pkl['sample_list']

    table_num = len(table_N_list)
    num_of_col = 0

    content_list = [] #[ [content1, content2], list of other tables ]
    filtered_sample_list = []
    used_table_id = []

    for i in range(table_num):
        tab_N = table_N_list[i]
        tab_D = table_D_list[i]
        tab_name = table_col_name[i]
        tab_type = table_types[i]
        tab_D = [k for k in tab_D if k!=0]
        tab_N = [k for k in tab_N if k!=0]
        tab_name, tab_type = filter_types(tab_name, tab_type)
        if len(tab_name) == len(tab_type) == len(tab_N) == len(tab_D) and len(common_prefix(tab_name)) < 2:
            tab_content = []
            tab_sample = []
            for j in range(len(tab_name)):
                num_of_col += 1
                if filter_names(tab_name[j]):
                    continue
                tab_content.append(f'{tab_name[j]}, {tab_type[j]}, {tab_N[j]}, {tab_D[j]}')
                tab_sample.append(sample_list[i][j])
            if len(tab_content) > 0:
                content_list.append(tab_content)
                used_table_id.append(i)
                filtered_sample_list.append(tab_sample)
    print(f'{num_of_col} cols in total.')

    save_dict = {
        'content_list': content_list,
        'used_table_id': used_table_id,
        'filtered_sample_list': filtered_sample_list,
    }

    with open(os.path.join(save_root_path, extracted_file_name), 'wb') as f:
        pickle.dump(save_dict, f)


if __name__ == '__main__':
    pkl_files = os.listdir(pickle_root_path)
    pkl_files = [i for i in pkl_files if i.endswith('.pkl')]

    print(f'{len(pkl_files)} files to be processed')

    executor = ProcessPoolExecutor(40)

    tasks = [executor.submit(col_level_data_process, pkl_files[i]) for i in range(len(pkl_files))]
    concurrent.futures.wait(tasks)

    print('All files are saved!')