import os
import sys
import ray
import argparse
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_columns', 500)

sys.path.append(os.getcwd())

from src.Attacker import Attacker
from src.ExptManager import ExptManager
from src.configs.default import *
from src.my_utils import *
from src.my_utils import _init_ray
from src.evaluation.eval_utils import *


def get_expt_manager_config():
    final_evaluator_config = get_final_evaluator_default_config()
    defender_config = get_defender_default_config()
    attacker_config = get_attacker_default_config()

    return {
        "dataset": "harmbench",
        "split": "val",
        "data_types": "standard",
        "is_wandb": False,
        "is_reload_attacks": False,
        "is_reload_completions": False,
        "base_save_path": "results/wildteaming/",
        "final_evaluator_config": final_evaluator_config,
        "defender_config": defender_config,
        "attacker_config": attacker_config}


def get_prune_labels(off_topics_pruner, low_risk_pruner):
    """
    Get the overall prune labels based on off_topics_pruner and low_risk_pruner.
    """
    return [(1 - int(ot == 0 and lr == 0)) for ot, lr in zip(off_topics_pruner, low_risk_pruner)]


def main(args, is_save=True):
    # need to use ray to manage loading multiple vllm models on the same machine
    expt_manager_config = get_expt_manager_config()
    _init_ray()

    ####### Set Custom Configs ########
    expt_manager_config["attacker_config"]["num_attacks"] = args.num_attacks
    expt_manager_config["attacker_config"]["num_tactics_per_attack"] = args.num_tactics_per_attack
    expt_manager_config["attacker_config"]["attacker_type"] = args.attacker_type

    print("=" * 20, "Configs", "=" * 20)
    print(expt_manager_config)
    print("=" * 50)

    print("Initializing...")
    expt_manager = ExptManager(expt_manager_config)
    test_cases = expt_manager.get_test_cases()
    print("ExptManager initialized")

    attacker = Attacker(attacker_config=expt_manager_config["attacker_config"])
    print("Attacker initialized")

    off_topics_pruner = get_pruner("wanli")
    low_risk_pruner = get_pruner("ai2_safety_request")
    print("Pruners initialized")

    all_behaviors = list(test_cases.keys())
    num_attacks = expt_manager_config["attacker_config"]["num_attacks"]
    behavior_target_map = expt_manager.get_behavior_target_map()

    all_attacks = {}
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)
        if expt_manager_config["attacker_config"]["attacker_type"] == "fix_lead_seed_sentence":
            behavior_id = test_cases[behavior]["behavior_info"]["BehaviorID"]
            behavior_target = behavior_target_map[behavior_id]
        else:
            behavior_target = None

        all_behavior_raw_attacks = []
        all_behavior_attacks = []
        all_behavior_attacks_tactics = []
        all_behavior_attacks_off_topics_prune_labels = []
        all_behavior_attacks_low_risk_prune_labels = []
        all_behavior_attacks_prune_labels = []
        num_valid_attacks = 0
        # generate attacks until we have enough valid attacks
        while (num_valid_attacks < num_attacks):
            batch_behavior_raw_attacks, batch_behavior_attacks, batch_behavior_attacks_tactics = attacker.get_attacks(behavior,
                                                                                                                    (num_attacks),
                                                                                                                    behavior_target)

            batch_off_topics_prune_labels = off_topics_pruner.prune_off_topics(behavior, batch_behavior_attacks)[0]
            batch_low_risk_prune_labels = low_risk_pruner.prune_low_risk(all_behavior_attacks)[0]
            batch_prune_labels = get_prune_labels(batch_off_topics_prune_labels, batch_low_risk_prune_labels)

            all_behavior_raw_attacks.extend(batch_behavior_raw_attacks)
            all_behavior_attacks.extend(batch_behavior_attacks)
            all_behavior_attacks_tactics.extend(batch_behavior_attacks_tactics)
            all_behavior_attacks_off_topics_prune_labels.extend(batch_off_topics_prune_labels)
            all_behavior_attacks_low_risk_prune_labels.extend(batch_low_risk_prune_labels)
            all_behavior_attacks_prune_labels.extend(batch_prune_labels)

            num_valid_attacks = len(all_behavior_attacks_prune_labels) - sum(all_behavior_attacks_prune_labels)
            print(f"Num Valid / All Attacks: {num_valid_attacks} / {len(all_behavior_attacks)}")

        all_attacks[behavior] = {"attacks": all_behavior_attacks,
                                 "raw_attacks": all_behavior_raw_attacks,
                                 "tactics": all_behavior_attacks_tactics,
                                 "off_topics_prune_labels": all_behavior_attacks_off_topics_prune_labels,
                                 "low_risk_prune_labels": all_behavior_attacks_low_risk_prune_labels,
                                 "prune_labels": all_behavior_attacks_prune_labels}
                                 

        if is_save:
            expt_manager.save_attacks(all_attacks)
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate wildteaming attacks')
    parser.add_argument('--num_attacks', type=int, default=10)
    parser.add_argument('--attacker_type', type=str, default="fix_lead_seed_sentence")
    parser.add_argument('--num_tactics_per_attack', type=int, default=3)
    args = parser.parse_args()

    main(args)
