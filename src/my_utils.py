from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import torch
from tqdm.auto import tqdm
import collections
import argparse

T = TypeVar('T')

import json
import yaml
import pandas as pd
import os
import ray
import random
from transformers.dynamic_module_utils import init_hf_modules


def _init_ray(num_cpus=32):
    # Start RAY
    # config different ports for ray head and ray workers to avoid conflict when running multiple jobs on one machine/cluster
    # docs: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats
    num_cpus = min([os.cpu_count(), num_cpus])

    os.environ['RAY_DEDUP_LOGS'] = '0'

    RAY_PORT = random.randint(0, 999) + 6000  # Random port in 6xxx zone
    RAY_MIN_PORT = random.randint(0, 489) * 100 + 10002
    RAY_MAX_PORT = RAY_MIN_PORT + 99  # Random port ranges zone

    os.environ['RAY_ADDRESS'] = f"127.0.0.1:{RAY_PORT}"
    ray_start_command = f"ray start --head --num-cpus={num_cpus} --port {RAY_PORT} --min-worker-port={RAY_MIN_PORT} --max-worker-port={RAY_MAX_PORT} --disable-usage-stats --include-dashboard=False"

    print(f"Starting Ray with command: {ray_start_command}")
    os.system(ray_start_command)

    init_hf_modules()  # Needed to avoid import error: https://github.com/vllm-project/vllm/pull/871
    ray.init(ignore_reinit_error=True)


def pretty_print_dict(d, indent=4):
    print(json.dumps(d, indent=indent))


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_open_instruct_training_data(output_path, prompts, completions, data_ids=None, dataset_name=""):
    if len(prompts) != len(completions):
        raise ValueError("Prompts and completions must have the same length.")

    if data_ids is None:
        data_ids = list(range(len(prompts)))
        if dataset_name != "":
            data_ids = [f"{x}--{dataset_name}" for x in data_ids]
    else:
        if len(data_ids) != len(prompts):
            raise ValueError("Data IDs must have the same length as prompts and completions.")
        else:
            if dataset_name != "":
                data_ids = [f"{x}--{dataset_name}" for x in data_ids]

    data = []
    for data_id, prompt, completion in zip(data_ids, prompts, completions):
        data.append({
            "dataset": dataset_name,
            "id": data_id,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": completion}],
        })

    ensure_dir(output_path)
    write_standard_data(data, output_path)


def load_standard_data(input_path: str, is_print=True) -> list[dict]:
    if ".jsonl" in input_path:
        with open(input_path, "r") as f:
            data = [json.loads(x) for x in f.readlines()]

        if is_print:
            print("=" * 50, "Example Data", "=" * 50)
            print(data[0])
            print("=" * 100)
            print(f"Loaded {len(data)} data points from {input_path}")
    elif ".json" in input_path:
        with open(input_path, "r") as f:
            data = json.load(f)
            for id, item in enumerate(data):
                if "id" not in item:
                    item["id"] = id
    elif ".tsv" in input_path:
        data = pd.read_csv(input_path, sep="\t").to_dict(orient="records")
        for id, item in enumerate(data):
            if "prompt" not in item and "attack" in item:
                item["prompt"] = item["attack"]
                if "id" not in item:
                    item["id"] = id
    else:
        raise ValueError(f"Unrecognized file type: {input_path}")
    return data


def write_standard_data(data: list[dict], output_path: str, makedirs: bool = True, is_print=True):
    if makedirs:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    if is_print:
        print(f"Saved {len(data)} data points to {output_path}")


def save_json(data: dict, output_path: str, makedirs: bool = True):
    if makedirs:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved json to {output_path}")


def read_json(input_path: str):
    with open(input_path, "r") as f:
        return json.load(f)


def export_standard_to_csv(data: list[dict], output_path: str):
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def import_csv_to_standard(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    data = []
    for i, row in df.iterrows():
        data.append(dict(row))
    return data


def convert_dict_to_namespace(d):
    """Convert dictionary to namespace."""
    namespace = argparse.Namespace()
    for k, v in d.items():
        setattr(namespace, k, v)
    return namespace


def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask):
    return value * mask + NEGATIVE_INF * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True):
    mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, '', flat)
    return flat


def distinctness(generations):
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])

    return len(unigrams) / total_words, len(bigrams) / total_words, len(trigrams) / total_words


def ensure_dir(d):
    d = d if os.path.isdir(d) else os.path.dirname(d)
    if not os.path.exists(d):
        os.makedirs(d)


def is_file_exist(file_path):
    return os.path.exists(file_path)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)
