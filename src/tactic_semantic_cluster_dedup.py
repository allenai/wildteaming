from fire import Fire
from typing import Callable
import numpy as np
from copy import deepcopy
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection

from src.tactics_utils import *


def cluster_and_dedup_items(
        items: list[str],
        model: SentenceTransformer,
        clustering_threshold: float,
        min_cluster_size: int,
        embed_batch_size: int
) -> tuple[list[list[int]], list[str]]:
    item_embeds = model.encode(
        items,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    clusters = community_detection(
        item_embeds,
        threshold=clustering_threshold,
        min_community_size=min_cluster_size,
        show_progress_bar=True
    )

    items_deduped = [items[cluster[0]] for cluster in clusters]
    return clusters, items_deduped


def cluster_and_dedup_with_getter(
        items: list[Any],
        data: dict,
        get_name_from_item: Callable[[Any], str],  # Function to get the name (top-level field of data) from an item
        get_repr_from_item: Callable[[Any], str],
        # Function to get the representation value to be embedded from an item
        get_cluster_data_from_item: Callable[[str], dict],  # Function to construct cluster data from an item
        model: SentenceTransformer,
        clustering_threshold: float,
        min_cluster_size: int,
        embed_batch_size: int
) -> tuple[list[dict], dict]:
    """
    Clusters a list of items by embedding textual representations of each item 
    with a SentenceTransformer model, running community detection on the embeddings
    to group similar items, and returns deduplicated clusters and data.

    Items are deduplicated by selecting a representative item for each cluster, 
    and merging data from items in the same cluster under the representative's name.

    Args:
    items: List of items to cluster and deduplicate.
    data: Dict mapping item names to per-item data.
    get_name_from_item: Fn to get the name for an item.
    get_repr_from_item: Fn to get the textual representation of an item.
    get_cluster_data_from_item: Fn to get per-item data for clusters.
    model: SentenceTransformer model.
    clustering_threshold: Community detection threshold.
    min_cluster_size: Minimum cluster size.
    embed_batch_size: Batch size for embedding.
  
    Returns:
    cluster_data: List of dicts with cluster info.
    data_deduped: Deduplicated data dict.
    """
    data = deepcopy(data)

    clusters, deduped = cluster_and_dedup_items(
        [get_repr_from_item(item) for item in items],
        model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    cluster_data = [
        {
            "size": len(cluster),
            "items": [
                get_cluster_data_from_item(items[idx]) for idx in cluster
            ]
        }
        for cluster in clusters
    ]

    data_deduped = defaultdict(lambda: defaultdict(list))
    clustered_indices = set()
    for cluster in clusters:
        # Sanity check that no item is in more than one cluster
        assert not any(idx in clustered_indices for idx in cluster), "Item found in multiple clusters"
        clustered_indices.update(cluster)

        representative_name = get_name_from_item(items[cluster[0]])
        for idx in cluster:
            deduped_name = get_name_from_item(items[idx])
            for field, values in data[deduped_name].items():
                data_deduped[representative_name][field] += values
    # Add in items that were not part of any clusters
    for i, item in enumerate(items):
        if i not in clustered_indices:
            name = get_name_from_item(item)
            data_deduped[name] = data[name]

    # Sanity check that the total number of inner fields is the same
    original_counts = defaultdict(int)
    for name in data:
        for field in data[name]:
            original_counts[field] += len(data[name][field])
    new_counts = defaultdict(int)
    for name in data_deduped:
        for field in data_deduped[name]:
            new_counts[field] += len(data_deduped[name][field])
    assert all(original_counts[field] == new_counts[field] for field in
               original_counts), "Found different amount of data between original and deduped"

    return cluster_data, data_deduped


def dedup_w_clustering(
        input_path: str,
        output_dir: str,
        clustering_threshold: float = 0.75,
        min_cluster_size: int = 10,
        embed_model_name: str = "nomic-ai/nomic-embed-text-v1",
        embed_batch_size: int = 64,
        seed: int = 42,
):
    args = dict(
        input_path=os.path.abspath(input_path),
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_model_name=embed_model_name,
        seed=seed
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/run_args.json', 'w') as f:
        json.dump(args, f, indent=2)

    rng = np.random.default_rng(seed=seed)
    with open(input_path, 'r') as f:
        data = json.load(f)
    model = SentenceTransformer(embed_model_name, trust_remote_code=True)

    ######### Cluster and dedup based on tactic name #########
    tactic_names: list[str] = list(data.keys())
    tactic_name_cluster_data, data_tactic_names_deduped = cluster_and_dedup_with_getter(
        tactic_names,
        data,
        get_name_from_item=lambda x: x,
        get_repr_from_item=lambda x: x,
        get_cluster_data_from_item=lambda x: dict(name=x),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    for i in range(len(tactic_name_cluster_data)):
        tactic_name_cluster_data[i]["names"] = [item["name"] for item in tactic_name_cluster_data[i]["items"]]

    print(
        f'Found {len(tactic_name_cluster_data)} clusters on names '
        f'with mean size of {np.mean([x["size"] for x in tactic_name_cluster_data])}'
    )

    # get_final_dedup(tactic_name_cluster_data,
    #                 f'{output_dir}/tactic_name_clusters_clean.json')

    # write_standard_data(tactic_name_cluster_data, f'{output_dir}/tactic_name_clusters.jsonl')
    with open(f'{output_dir}/deduped_tactic_name.json', 'w') as f:
        json.dump(data_tactic_names_deduped, f, indent=2)

    ######### Cluster and dedup based on tactic description #########
    names_and_defs = [(name, rng.choice(tactic["definition"])) for name, tactic in
                      data.items()]  # Track reverse mapping
    tactic_definition_cluster_data, data_tactic_definitions_deduped = cluster_and_dedup_with_getter(
        names_and_defs,
        data,
        get_name_from_item=lambda item: item[0],
        get_repr_from_item=lambda item: item[1],
        get_cluster_data_from_item=lambda item: dict(name=item[0], definition=item[1]),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    for i in range(len(tactic_definition_cluster_data)):
        tactic_definition_cluster_data[i]["names"] = [item["name"] for item in
                                                      tactic_definition_cluster_data[i]["items"]]
        tactic_definition_cluster_data[i]["definitions"] = [item["definition"] for item in
                                                            tactic_definition_cluster_data[i]["items"]]

    print(
        f'Found {len(tactic_definition_cluster_data)} clusters on definitions '
        f'with mean size of {np.mean([x["size"] for x in tactic_definition_cluster_data])}'
    )

    get_final_dedup(tactic_definition_cluster_data,
                    f'{output_dir}/tactic_definition_clusters_clean.json')

    write_standard_data(tactic_definition_cluster_data, f'{output_dir}/tactic_definition_clusters.jsonl')
    with open(f'{output_dir}/deduped_tactic_definition.json', 'w') as f:
        json.dump(data_tactic_definitions_deduped, f, indent=2)

    ######### Cluster and dedup based on both tactic name and description #########
    name_deduped_names_and_defs = [(name, rng.choice(tactic["definition"])) for name, tactic in
                                   data_tactic_names_deduped.items()]  # Track reverse mapping
    name_deduped_tactic_definition_cluster_data, name_deduped_data_tactic_definitions_deduped = cluster_and_dedup_with_getter(
        name_deduped_names_and_defs,
        data_tactic_names_deduped,
        get_name_from_item=lambda item: item[0],
        get_repr_from_item=lambda item: item[1],
        get_cluster_data_from_item=lambda item: dict(name=item[0], definition=item[1]),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    for i in range(len(name_deduped_tactic_definition_cluster_data)):
        name_deduped_tactic_definition_cluster_data[i]["names"] = [item["name"] for item in
                                                                   name_deduped_tactic_definition_cluster_data[i][
                                                                       "items"]]
        name_deduped_tactic_definition_cluster_data[i]["definitions"] = [item["definition"] for item in
                                                                         name_deduped_tactic_definition_cluster_data[i][
                                                                             "items"]]

    print(
        f'Found {len(name_deduped_tactic_definition_cluster_data)} clusters on definitions from deduped names '
        f'with mean size of {np.mean([x["size"] for x in name_deduped_tactic_definition_cluster_data])}'
    )

    get_final_dedup(name_deduped_tactic_definition_cluster_data,
                    f'{output_dir}/tactic_definition_clusters_from_deduped_names_clean.json')

    write_standard_data(name_deduped_tactic_definition_cluster_data,
                        f'{output_dir}/tactic_definition_clusters_from_deduped_names.jsonl')
    with open(f'{output_dir}/deduped_tactic_definition_from_deduped_names.json', 'w') as f:
        json.dump(name_deduped_data_tactic_definitions_deduped, f, indent=2)

    return tactic_name_cluster_data, tactic_definition_cluster_data, name_deduped_tactic_definition_cluster_data


def main(
        input_path: str,
        output_dir: str,
        clustering_threshold: float = 0.75,
        min_cluster_size: int = 10,
        embed_model_name: str = "nomic-ai/nomic-embed-text-v1",
        embed_batch_size: int = 64,
        seed: int = 42,
):
    args = dict(
        input_path=os.path.abspath(input_path),
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_model_name=embed_model_name,
        seed=seed
    )
    with open(f'{output_dir}/run_args.json', 'w') as f:
        json.dump(args, f, indent=2)

    rng = np.random.default_rng(seed=seed)
    with open(input_path, 'r') as f:
        data = json.load(f)
    model = SentenceTransformer(embed_model_name, trust_remote_code=True)

    # Cluster and dedup based on tactic name
    tactic_names: list[str] = list(data.keys())
    tactic_name_cluster_data, data_tactic_names_deduped = cluster_and_dedup_with_getter(
        tactic_names,
        data,
        get_name_from_item=lambda x: x,
        get_repr_from_item=lambda x: x,
        get_cluster_data_from_item=lambda x: dict(name=x),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )
    print(
        f'Found {len(tactic_name_cluster_data)} clusters on names '
        f'with mean size of {np.mean([x["size"] for x in tactic_name_cluster_data])}'
    )
    write_standard_data(tactic_name_cluster_data, f'{output_dir}/tactic_name_clusters.jsonl')

    with open(f'{output_dir}/deduped_tactic_name.json', 'w') as f:
        json.dump(data_tactic_names_deduped, f, indent=2)

    # Cluster and dedup based on tactic description
    names_and_defs = [(name, rng.choice(tactic["definition"])) for name, tactic in
                      data.items()]  # Track reverse mapping
    tactic_definition_cluster_data, data_tactic_definitions_deduped = cluster_and_dedup_with_getter(
        names_and_defs,
        data,
        get_name_from_item=lambda item: item[0],
        get_repr_from_item=lambda item: item[1],
        get_cluster_data_from_item=lambda item: dict(name=item[0], definition=item[1]),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    print(
        f'Found {len(tactic_definition_cluster_data)} clusters on definitions '
        f'with mean size of {np.mean([x["size"] for x in tactic_definition_cluster_data])}'
    )

    write_standard_data(tactic_definition_cluster_data, f'{output_dir}/tactic_definition_clusters.jsonl')
    with open(f'{output_dir}/deduped_tactic_definition.json', 'w') as f:
        json.dump(data_tactic_definitions_deduped, f, indent=2)

    # Cluster and dedup based on both tactic name and description
    name_deduped_names_and_defs = [(name, rng.choice(tactic["definition"])) for name, tactic in
                                   data_tactic_names_deduped.items()]  # Track reverse mapping
    name_deduped_tactic_definition_cluster_data, name_deduped_data_tactic_definitions_deduped = cluster_and_dedup_with_getter(
        name_deduped_names_and_defs,
        data_tactic_names_deduped,
        get_name_from_item=lambda item: item[0],
        get_repr_from_item=lambda item: item[1],
        get_cluster_data_from_item=lambda item: dict(name=item[0], definition=item[1]),
        model=model,
        clustering_threshold=clustering_threshold,
        min_cluster_size=min_cluster_size,
        embed_batch_size=embed_batch_size
    )

    print(
        f'Found {len(name_deduped_tactic_definition_cluster_data)} clusters on definitions from deduped names '
        f'with mean size of {np.mean([x["size"] for x in name_deduped_tactic_definition_cluster_data])}'
    )

    write_standard_data(name_deduped_tactic_definition_cluster_data,
                        f'{output_dir}/tactic_definition_clusters_from_deduped_names.jsonl')
    with open(f'{output_dir}/deduped_tactic_definition_from_deduped_names.json', 'w') as f:
        json.dump(name_deduped_data_tactic_definitions_deduped, f, indent=2)


if __name__ == '__main__':
    Fire(main)
