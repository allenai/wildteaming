import os
import sys
import ray
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_columns', 500)

sys.path.append(os.getcwd())

from src.Attacker import Attacker
from src.Pruner import Pruner
from src.ExptManager import ExptManager
from src.utils import *
from src.configs.default import *


def get_expt_manager_config():
    final_evaluator_config = get_final_evaluator_default_config()
    defender_config = get_defender_default_config()
    attacker_config = get_attacker_default_config()
    pruner_config = get_pruner_default_config()

    return {
            "dataset": "harmbench",
            "split": "test",  # "val"
            "data_types": "standard",  # "all",  # ["standard", "contextual"]
            "is_wandb": False,
            "is_reload_attacks": False,
            "is_reload_completions": False,
            "is_prune_off_topics": True,
            "base_save_path": "results/wildteaming/",
            "final_evaluator_config": final_evaluator_config,
            "defender_config": defender_config,
            "attacker_config": attacker_config,
            "pruner_config": pruner_config}


def main(args, is_save=True):
    # need to use ray to manage loading multiple vllm models on the same machine
    expt_manager_config = get_expt_manager_config()
    _init_ray()

    ####### Set Custom Configs ########
    expt_manager_config["attacker_config"]["num_attacks"] = args.num_attacks
    expt_manager_config["attacker_config"]["num_tactics_per_attack"] = args.num_tactics_per_attack
    expt_manager_config["attacker_config"]["attacker_type"] = args.attacker_type

    print("Initializing...")
    expt_manager = ExptManager(expt_manager_config)
    test_cases = expt_manager.get_test_cases()
    print("ExptManager initialized")

    print("=" * 20, "Configs", "=" * 20)
    print(expt_manager_config)
    print("=" * 50)

    attacker = Attacker(attacker_config=expt_manager_config["attacker_config"])
    print("Attacker initialized")

    if expt_manager_config["is_prune_off_topics"]:
        pruner = Pruner(pruner_config=expt_manager_config["pruner_config"])
        print("Pruner initialized")

    all_behaviors = list(test_cases.keys())
    num_attacks = expt_manager_config["attacker_config"]["num_attacks"]
    behavior_target_map = expt_manager.get_behavior_target_map()

    all_attacks = {}
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)
        if expt_manager_config["attacker_config"]["attacker_type"] == "fix_prefix_injection":
            behavior_id = test_cases[behavior]["behavior_info"]["BehaviorID"]
            behavior_target = behavior_target_map[behavior_id]
        else:
            behavior_target = None

        all_behavior_attacks = []
        all_behavior_raw_attacks = []
        all_behavior_attacks_tactics = []
        all_behavior_attacks_prune_labels = []
        all_behavior_attacks_prune_text_labels = []
        while (len(all_behavior_attacks_prune_labels) - sum(all_behavior_attacks_prune_labels)) < num_attacks:
            behavior_raw_attacks, behavior_attacks, behavior_tactics = attacker.get_attacks(behavior,
                                                                                            (num_attacks // 2),
                                                                                            behavior_target)
            prune_labels, prune_text_labels, probs_list = pruner.prune_off_topics(behavior, behavior_attacks)

            all_behavior_attacks.extend(behavior_attacks)
            all_behavior_raw_attacks.extend(behavior_raw_attacks)
            all_behavior_attacks_tactics.extend(behavior_tactics)
            all_behavior_attacks_prune_labels.extend(prune_labels)
            all_behavior_attacks_prune_text_labels.extend(prune_text_labels)

        print("Total attacks:", len(all_behavior_attacks))

        all_attacks[behavior] = {"attacks": all_behavior_attacks,
                                 "raw_attacks": all_behavior_raw_attacks,
                                 "tactics": all_behavior_attacks_tactics,
                                 "prune_labels": all_behavior_attacks_prune_labels,
                                 "prune_text_labels": all_behavior_attacks_prune_text_labels}
        if is_save:
            expt_manager.save_attacks(all_attacks)
    ray.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate model completions for wildteaming')
    parser.add_argument('--attacker_type', type=str, default="fix_prefix_injection")
    args = parser.parse_args()

    args = convert_dict_to_namespace({"num_attacks": 2,
                                      "num_tactics_per_attack": 4,
                                      "attacker_type": args.attacker_type})

