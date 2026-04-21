"""quickstart_xgboost: A Flower / XGBoost app."""

import numpy as np
import os
import xgboost as xgb
from flwr_datasets import FederatedDataset
from dotenv import load_dotenv
from flwr_datasets.partitioner import IidPartitioner

load_dotenv() 

#listing the possible kmers
KMER_COLS = [
    "aaa","aac","aag","aat","aca","acc","acg","act","aga","agc","agg","agt",
    "ata","atc","atg","att","caa","cac","cag","cat","cca","ccc","ccg","cct",
    "cga","cgc","cgg","cgt","cta","ctc","ctg","ctt","gaa","gac","gag","gat",
    "gca","gcc","gcg","gct","gga","ggc","ggg","ggt","gta","gtc","gtg","gtt",
    "taa","tac","tag","tat","tca","tcc","tcg","tct","tga","tgc","tgg","tgt",
    "tta","ttc","ttg","ttt"
]
#for now our single predictive target which is predicting acetate growth resistance
LABEL_COL = "YPACETATE"

# no longer needed since we have made the split ourselves on hugging face database, but leaving in case we want to use in future

# def train_test_split(partition, test_fraction, seed):
#     """Split the data into train and validation set given split rate."""
#     train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
#     partition_train = train_test["train"]
#     partition_test = train_test["test"]

#     num_train = len(partition_train)
#     num_test = len(partition_test)

#     return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost. Will implement normalization here as well"""
    batch = data[:]
    x = np.asarray([batch[col] for col in KMER_COLS], dtype=np.float32).T
    sum_rows = x.sum(axis=1, keepdims=True)
    x_norm = np.divide(x, sum_rows, where=sum_rows != 0, out=np.zeros_like(x))  # Div by 0 handling divides where sum of rows is not 0
    y = np.asarray(batch[LABEL_COL], dtype=np.float32)
    return xgb.DMatrix(x_norm, label=y)


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    print(f"num_clients: {num_clients}, partition_id: {partition_id}")
    """Load partition Yeast k_mer dataset data."""
    # Only initialize `FederatedDataset` once
    global fds
    hugging_dataset = os.getenv("YEAST_DATASET")
    hf_token = os.getenv("HF_TOKEN")
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        
        fds = FederatedDataset(
            dataset=hugging_dataset,
            partitioners={"train": partitioner},
            data_files={
                    "train": "training_set.csv",
                    "test": "testing_set.csv",
                    "validation": "validation_set.csv"
            },
            token=hf_token,
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")

    # Train/test splitting
    valid_data = fds.load_split("validation")
    valid_data.set_format("numpy")

    num_train = len(partition)
    num_val = len(valid_data)

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(partition)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
