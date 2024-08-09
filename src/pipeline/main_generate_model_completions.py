import os
import sys
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_columns', 500)

sys.path.append(os.getcwd())

from src.Defender import Defender
from src.ExptManager import ExptManager
from src.my_utils import _init_ray
from src.configs.default import *
from src.evaluation.eval_utils import *


def get_expt_manager_config():
    final_evaluator_config = get_final_evaluator_default_config()
    defender_config = get_defender_default_config()
    attacker_config = get_attacker_default_config()

    return {"dataset": "harmbench",
            "split": "val",
            "data_types": "standard",
            "is_wandb": False,
            "is_reload_attacks": False,
            "is_reload_completions": False,
            "base_save_path": "results/wildteaming/",
            "final_evaluator_config": final_evaluator_config,
            "defender_config": defender_config,
            "attacker_config": attacker_config}


def set_defender_model(expt_manager_config, model_name):
    expt_manager_config["defender_config"]["model_name"] = model_name
    return expt_manager_config


def main(args, is_save=True):
    # need to use ray to manage loading multiple vllm models on the same machine
    expt_manager_config = get_expt_manager_config()
    _init_ray()

    ####### Set Custom Configs ########
    expt_manager_config["defender_config"]["model_name"] = args.model_name
    expt_manager_config["defender_config"]["n_devices"] = args.n_devices
    expt_manager_config["attacker_config"]["num_attacks"] = args.num_attacks
    expt_manager_config["attacker_config"]["attacker_type"] = args.attacker_type
    expt_manager_config["attacker_config"]["num_tactics_per_attack"] = args.num_tactics_per_attack

    print("=" * 20, "Configs", "=" * 20)
    print(expt_manager_config)
    print("=" * 80)

    print("Initializing...")
    expt_manager = ExptManager(expt_manager_config)
    test_cases = expt_manager.get_test_cases()
    print("ExptManager initialized")

    defender = Defender(defender_config=expt_manager_config["defender_config"])
    print("Defender initialized")

    all_completions = {}
    all_behaviors = list(test_cases.keys())
    num_attacks = expt_manager_config["attacker_config"]["num_attacks"]
    all_attacks = expt_manager.load_attacks()
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)
        
        all_behavior_attacks = all_attacks[behavior]["attacks"]
        all_behavior_attacks_prune_labels = all_attacks[behavior]["prune_labels"]

        all_behavior_attacks_selected = []
        for a, p in zip(all_behavior_attacks, all_behavior_attacks_prune_labels):
            if p == 0:
                all_behavior_attacks_selected.append(a)

        all_behavior_attacks_selected = all_behavior_attacks_selected[:num_attacks]
        behavior_completions = defender.get_model_completions(all_behavior_attacks_selected)

        all_completions[behavior] = behavior_completions

        if is_save:
            expt_manager.save_model_completions(all_completions)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate model completions')
    parser.add_argument('--model_name', type=str, default="lmsys/vicuna-7b-v1.5",
                        help='Model name to generate completions for')
    parser.add_argument('--num_attacks', type=int, default=10)
    parser.add_argument('--num_tactics_per_attack', type=int, default=3)
    parser.add_argument('--attacker_type', type=str, default="fix_lead_seed_sentence")
    parser.add_argument('--n_devices', type=int, default=2)
    parser.add_argument('--num_attacks_evaluated', type=int, default=2)
    args = parser.parse_args()

    main(args)
