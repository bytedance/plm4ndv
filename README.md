# Introduction

This repository contains the implementation of **PLM4NDV**. We provide details in our paper, including but not limited to the train/validation/test dataset splits, preprocessed data, semantic embedding, and model training. You can obtain the results presented in our paper by following the instructions below.

# Instruction

1. Establish the experimental environment in Python 3.10.

```bash
pip3 install -r requirement.txt
```

2. Download [TabLib](https://huggingface.co/datasets/approximatelabs/tablib-v1-sample) dataset, and put the parquet files in a folder.
3. Read the parquet files, extract primary component of each table from each file, and the extracted content should be saved to `./data/extracted/`. The default data access method is set as sequential access, if you want to use random sampling please comment out Line 56 and use Line 57 instead.

```bash
python extract_parquet.py
```

4. Traverse the extracted content, filter useless columns and save the filtered content to `./data/traversed/`.

```bash
python traverse_columns.py
```

5. Split the traversed content into train/test/val sets, deduplicate the contents and save to `./data/splitted/`.

```bash
python split_traversed.py
```

6. Download [sentence-t5-large](https://huggingface.co/sentence-transformers/sentence-t5-large) and set the model path. Generate the embedding of a column using PLM. Save them to `./data/embedding/`.

```bash
python semantic_embedding.py
```

7. Train the model and the model parameters will be saved to `./ckpt/`. The inference code is also in the file and the performance on NDV estimation under sequential access reproted in the paper will be presented.

```bash
python train_and_test.py
```

> If you want to reproduce the performance of out method under random sampling 100 rows, please follow the instructions in Step 3.
>
> If you do not want to train the model from scratch, you can load our model parameters to obtain the results on the test set by commenting out Line 300 in `train_and_test.py`.
