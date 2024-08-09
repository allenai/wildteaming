import pandas as pd
from tqdm import tqdm
import random
import string
import numpy as np

from src.my_utils import *


def load_existing_tactics():
    data_filename = "data/tactics/manual_tactics.tsv"
    df = pd.read_csv(data_filename, sep='\t')

    return df["strategy"].tolist(), df["definition"].tolist()


def get_final_dedup(data, data_path):
    # set random seed
    np.random.seed(42)

    tactic_clusters = {}
    for i, d in enumerate(data):
        tactic_name_list = d["names"]
        tactic_definition_list = d["definitions"]
        representative_name = random.choice(tactic_name_list)

        tactic_clusters[representative_name] = {}
        tactic_clusters[representative_name]["definitions"] = tactic_definition_list
        tactic_clusters[representative_name]["names"] = tactic_name_list
        tactic_clusters[representative_name]["size"] = len(tactic_name_list)
        tactic_clusters[representative_name]["cluster_id"] = i

    save_path = data_path.replace(".jsonl", "_clean.json")
    with open(save_path, "w") as f:
        json.dump(tactic_clusters, f, indent=2)


def get_strategy_map(df_data, save_path="data/tactics/auto_tactics_frequency.json", is_load_saved_map=True):
    if is_load_saved_map:
        with open(save_path, "r") as f:
            all_strategies_map = json.load(f)
        print("Loaded strategy map from", save_path)
    else:
        all_existing_strategies, all_existing_definitions = load_existing_tactics()

        all_strategies_map = {}
        for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
            strategy = row["strategy"].lower().strip()
            definition = row["definition"]
            excerpt = row["excerpt"]
            reason = row["reason"]
            strategy_type = row["strategy_type"]
            uid = row["uid"]

            if strategy_type == "existing_strategies":
                if strategy not in all_existing_strategies:
                    continue
                else:
                    definition = all_existing_definitions[all_existing_strategies.index(strategy)]

            if strategy not in all_strategies_map:
                all_strategies_map[strategy] = {"definition": [],
                                                "excerpt": [],
                                                "reason": [],
                                                "uid": [],
                                                "strategy_type": []}

            # print(all_strategies_map)
            # print("-")

            all_strategies_map[strategy]["definition"].append(definition)
            all_strategies_map[strategy]["excerpt"].append(excerpt)
            all_strategies_map[strategy]["reason"].append(reason)
            all_strategies_map[strategy]["uid"].append(uid)
            all_strategies_map[strategy]["strategy_type"].append(strategy_type)

        # save json
        with open(save_path, "w") as f:
            json.dump(all_strategies_map, f)
        print("Saved strategy map to", save_path)

    return all_strategies_map


def parse_existing_strategy_list(user_uttr_strategy_existing):
    user_uttr_strategy_existing_parsed = user_uttr_strategy_existing.split("- ")
    user_uttr_strategy_existing_parsed = user_uttr_strategy_existing_parsed[1:]

    all_parsed = {"strategy": [], "excerpt": [], "reason": []}
    for e in user_uttr_strategy_existing_parsed:
        e_strategy = None
        e_excerpt = None
        e_reason = None
        try:
            e = e.replace("Excerpt", "excerpt")
            e = e.replace("exerpt", "excerpt")
            e = e.replace("  ", " ")
            e = e.replace("Reason", "reason")

            e_list = e.split(": [excerpt]")
            e_strategy = e_list[0]
            excerpt = e_list[1].split("[reason]")
            e_excerpt = excerpt[0]
            e_reason = excerpt[1]

            e_strategy = e_strategy.lower().replace("\n", " ").strip()
            e_excerpt = e_excerpt.lower().strip()
            e_reason = e_reason.lower().replace("\n", " ").strip()
        except:
            continue

        if e_strategy is None and e_excerpt is None and e_reason is None:
            continue
        else:
            all_parsed["strategy"].append(e_strategy)
            all_parsed["excerpt"].append(e_excerpt)
            all_parsed["reason"].append(e_reason)

    return all_parsed


def parse_new_strategy_list(user_uttr_strategy_new):
    user_uttr_strategy_new_parsed = user_uttr_strategy_new.split("- ")
    user_uttr_strategy_new_parsed = user_uttr_strategy_new_parsed[1:]

    all_parsed = {"strategy": [], "definition": [], "excerpt": [], "reason": []}
    for e in user_uttr_strategy_new_parsed:
        e_strategy = None
        e_definition = None
        e_excerpt = None
        e_reason = None
        try:
            e = e.replace("Excerpt", "excerpt")
            e = e.replace("exerpt", "excerpt")
            e = e.replace("  ", " ")
            e = e.replace("Reason", "reason")
            e = e.replace("Definition", "definition")

            e_list = e.split(": [excerpt]")
            e_strategy = e_list[0]

            if "(" in e_strategy and ")" in e_strategy:
                e_strategy_list = e_strategy.split("(")
                e_strategy = e_strategy_list[0]
                e_definition = e_strategy_list[1].split(")")[0]

            excerpt = e_list[1].split("[reason]")
            e_excerpt = excerpt[0]
            e_reason = excerpt[1]

            e_strategy = e_strategy.lower().replace("\n", " ").strip()
            e_definition = e_definition.lower().replace("\n", " ").strip()
            e_excerpt = e_excerpt.lower().strip()
            e_reason = e_reason.lower().replace("\n", " ").strip()
        except:
            continue

        if e_strategy is None and e_definition is None and e_excerpt is None and e_reason is None:
            continue
        else:
            all_parsed["strategy"].append(e_strategy)
            all_parsed["definition"].append(e_definition)
            all_parsed["excerpt"].append(e_excerpt)
            all_parsed["reason"].append(e_reason)

    return all_parsed


def parse_raw_strategies(df_data):
    all_parsed_data = []
    for index, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Parsing strategies"):
        row = row.to_dict()
        parsed_data = {}

        for c in row:
            parsed_data[c] = row[c]

        user_uttr = row["user_uttr"]
        user_uttr_simp = row["user_uttr_simp"]
        user_uttr_strategy = row["user_uttr_strategy"]
        user_uttr_strategy = "- " + user_uttr_strategy

        if "*New strategies that are not in the existing list:*" in user_uttr_strategy:
            try:
                user_uttr_strategy = user_uttr_strategy.split("*New strategies that are not in the existing list:*")
                user_uttr_strategy_existing = user_uttr_strategy[0]
                user_uttr_strategy_new = user_uttr_strategy[1]
            except:
                print("BAD format; skip!")
                continue
        else:
            user_uttr_strategy_existing = user_uttr_strategy
            user_uttr_strategy_new = ""

        user_uttr_strategy_existing_parsed = parse_existing_strategy_list(user_uttr_strategy_existing)
        user_uttr_strategy_new_parsed = parse_new_strategy_list(user_uttr_strategy_new)

        parsed_data["existing_strategies"] = user_uttr_strategy_existing_parsed
        parsed_data["new_strategies"] = user_uttr_strategy_new_parsed

        all_parsed_data.append(parsed_data)

    print(len(all_parsed_data))
    return all_parsed_data


def random_string(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
