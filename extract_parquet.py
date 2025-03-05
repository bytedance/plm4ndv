# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import os
import pyarrow as pa
import pandas as pd
import numpy as np
import pickle
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def makedirs(path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def load_parquet_file(file_path):
    assert file_path.endswith('.parquet')
    df = pd.read_parquet(file_path)
    tables = [pa.RecordBatchStreamReader(b).read_all() for b in df['arrow_bytes']]
    return tables

current_path = os.path.dirname(os.path.abspath(__file__))
# ! set the dataset path
parquet_root_path = os.path.join(current_path, '../tablib-sample')


def extract_parquet(file_name):
    print(f'extract {file_name}...')
    save_root_path = os.path.join(current_path, 'data/extracted')
    makedirs(save_root_path)

    tables = load_parquet_file(os.path.join(parquet_root_path, file_name))

    table_N_list = []
    table_D_list = []
    table_col_name = []
    table_types = []
    sample_list = []
    used_table_idx = []
    for j, pyarrow_table in enumerate(tables):
        N_list = []
        D_list = []
        name_list = []
        type_list = []
        samples = []
        col_num = len(pyarrow_table.column_names)
        for i in range(col_num):
            col_data = pyarrow_table.column(i)
            try:
                col = pa.array(col_data).drop_null().to_pylist() # []
                N = len(col)
                if N < 10000:
                    continue
                D = len(set(col))
                N_list.append(N)
                D_list.append(D)
                samples.append(col[:100]) # select the top 100 rows
                # samples.append(np.random.choice(col, size=100, replace=True).tolist())# randomly select 100 rows
            except Exception as e:
                print('ERROR', e)
        if len(N_list) == 0:
            continue
        table_N_list.append(N_list)
        table_D_list.append(D_list)
        table_col_name.append(pyarrow_table.column_names)
        table_types.append(pyarrow_table.schema.types)
        sample_list.append(samples)
        used_table_idx.append(j)
    save_dict = {
        'table_num': len(tables),
        'table_N_list': table_N_list,
        'table_D_list': table_D_list,
        'table_col_name': table_col_name,
        'table_types': table_types,
        'sample_list': sample_list,
        'used_table_idx': used_table_idx
    }
    file_id = file_name.split('.')[0]

    save_path = os.path.join(save_root_path, f'{file_id}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f'save at {save_path}')


if __name__ == '__main__':
    t0 = time.time()

    parquet_files = os.listdir(parquet_root_path)
    parquet_files = [i for i in parquet_files if i.endswith('.parquet')]
    
    # ignore three files due to memory issue
    parquet_files = [i for i in parquet_files if i.split('.')[0] not in ['2d7d54b8', '8e1450ee', 'dc0e820c']]
    print(f'{len(parquet_files)} files to be processed')
    
    executor = ProcessPoolExecutor(40)

    tasks = [executor.submit(extract_parquet, parquet_files[i]) for i in range(len(parquet_files))]
    concurrent.futures.wait(tasks)

    print('All files are saved!')
    t1 = time.time()
    cost = t1 - t0
    print(f'cost time: {cost:.3f} s')