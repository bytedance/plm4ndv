import os
from extract_parquet import makedirs
import pickle
from sentence_transformers import SentenceTransformer
import torch
from collections import Counter

current_path = os.path.dirname(os.path.abspath(__file__))

split_root_path = os.path.join(current_path, 'data/split_table')
save_root_path = os.path.join(current_path, f'data/embedding')
makedirs(save_root_path)
# ! set PLM path
model_dir = os.path.join(current_path, '../huggingface/sentence_transformer/sentence-t5-large')
plm_model = SentenceTransformer(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_column_profile(data):
    """
    profile: f_j, 1 <= j <= n, n = len(all_sampled_data)
    """
    value_counts = Counter(data)
    data_len = len(data)
    freq = [0] * (data_len + 1)
    for value, count in value_counts.items():
        freq[count] += 1
    return freq

def gen_embeddings(file_name):
    file_path = os.path.join(split_root_path, file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    contents = data['content']
    sample_list = data['sample_list']

    strings = []
    N = []
    D = []
    profile = []
    name_list = []
    type_list = []

    tab_cols_cnt = [] 
    for i, tab_content in enumerate(contents):
        num_of_cols = len(tab_content)
        tab_cols_cnt.append(num_of_cols)

        for j, col_content in enumerate(tab_content):
            content_splited = col_content.split(',')
            name_list.append(content_splited[:-3])
            type_list.append(content_splited[-3])
            N.append(int(content_splited[-2]))
            D.append(int(content_splited[-1]))
            col_description = ','.join(content_splited[:-2])
            
            strings.append(col_description)
            profile.append(build_column_profile(sample_list[i][j]))
    
    print(f'data num: {len(strings)}')
    with torch.no_grad():
        embeddings = plm_model.encode(strings, device=device)
    
    save_dict = {
        'name_list': name_list,
        'type_list': type_list,
        'embeddings': embeddings,
        'N': N,
        'D': D,
        'profile': profile,
        'tab_cols_cnt': tab_cols_cnt,
    }
    save_path = os.path.join(save_root_path, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f'saved {save_path}')


if __name__ == '__main__':
    gen_embeddings('train.pkl')
    gen_embeddings('test.pkl')
    gen_embeddings('val.pkl')