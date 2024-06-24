import os
import sys
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_columns', 500)

sys.path.append(os.getcwd())

from src.jailbreak_baselines.wildteaming.Defender import Defender
from src.jailbreak_baselines.wildteaming.ExptManager import ExptManager
from src.jailbreak_baselines.wildteaming.utils import _init_ray
from src.jailbreak_baselines.wildteaming.configs.default import *
from src.jailbreak_baselines.evaluation.eval_utils import *


def get_expt_manager_config():
    final_evaluator_config = get_final_evaluator_default_config()
    intermediate_evaluators_config = get_intermediate_evaluators_default_config()
    defender_config = get_defender_default_config()
    attacker_config = get_attacker_default_config()
    pruner_config = get_pruner_default_config()

    return {"dataset": "harmbench",
            "split": "test",
            "data_types": "standard",  # "all",  # ["standard", "contextual"]
            "is_wandb": False,
            "is_reload_attacks": False,
            "is_reload_completions": False,
            "is_prune_off_topics": True,
            "base_save_path": "results/wildteaming/",
            "final_evaluator_config": final_evaluator_config,
            "intermediate_evaluators_config": intermediate_evaluators_config,
            "defender_config": defender_config,
            "attacker_config": attacker_config,
            "pruner_config": pruner_config}


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

    low_risk_pruner = get_pruner("ai2_safety_request")
    off_topic_pruner = get_pruner("wanli")

    all_behaviors = list(test_cases.keys())
    num_attacks = expt_manager_config["attacker_config"]["num_attacks"]
    all_attacks = expt_manager.load_attacks()

    all_completions = expt_manager.load_model_completions()
    if all_completions is None:
        all_completions = {}

    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)

        # TODO to fix the reloading
        if behavior in all_completions:
            print(f"Skipping {behavior} as completions already exist.")
            continue

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


def main_all_prune(args, is_save=True):
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

    low_risk_pruner = get_pruner("ai2_safety_request")
    llamaguard2_low_risk_pruner = get_pruner("llamaguard2")

    all_behaviors = list(test_cases.keys())
    all_attacks = expt_manager.load_attacks()

    all_completions = {}
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)

        all_behavior_attacks = all_attacks[behavior]["attacks"]
        all_behavior_attacks_off_topics_prune_labels = all_attacks[behavior]["prune_labels"]
        all_behavior_attacks_low_risk_prune_labels = low_risk_pruner.prune_low_risk(all_behavior_attacks)[0]
        all_behavior_attacks_llamaguard2_prune_labels = \
        llamaguard2_low_risk_pruner.prune_low_risk(all_behavior_attacks)[0]
        all_behavior_completions = defender.get_model_completions(all_behavior_attacks)

        all_completions[behavior] = {"completions": all_behavior_completions,
                                     "attacks": all_behavior_attacks,
                                     "nli_prune_labels": all_behavior_attacks_off_topics_prune_labels,
                                     "low_risk_prune_labels": all_behavior_attacks_low_risk_prune_labels,
                                     "llamaguard2_prune_labels": all_behavior_attacks_llamaguard2_prune_labels}
        if is_save:
            expt_manager.save_model_completions(all_completions)


def main_test_set(args, is_save=True):
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

    all_behaviors = list(test_cases.keys())
    all_attacks = expt_manager.load_attacks()

    all_completions = {}
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)

        all_behavior_attacks = all_attacks[behavior]["attacks"]
        all_behavior_attacks_off_topics_prune_labels = all_attacks[behavior]["off_topics_prune_labels"]
        all_behavior_attacks_low_risk_prune_labels = all_attacks[behavior]["low_risk_prune_labels"]

        selected_idx = [i for i in range(len(all_behavior_attacks)) if
                        all_behavior_attacks_off_topics_prune_labels[i] == 0 and
                        all_behavior_attacks_low_risk_prune_labels[i] == 0][:args.num_attacks_evaluated]

        all_behavior_attacks = [all_behavior_attacks[i] for i in selected_idx]
        if len(all_behavior_attacks) < args.num_attacks_evaluated:
            print(f"Not enough attacks for {behavior}")
            return
        all_behavior_completions = defender.get_model_completions(all_behavior_attacks)
        print("Number of attacks: ", len(all_behavior_attacks))
        print("Number of completions: ", len(all_behavior_completions))

        all_completions[behavior] = {"completions": all_behavior_completions,
                                     "attacks": all_behavior_attacks}
        if is_save:
            expt_manager.save_model_completions(all_completions)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate model completions for wildteaming')
    parser.add_argument('--model_name', type=str, default="lmsys/vicuna-7b-v1.5",
                        help='Model name to generate completions for')
    parser.add_argument('--num_attacks', type=int, default=100)
    parser.add_argument('--num_tactics_per_attack', type=int, default=3)
    parser.add_argument('--attacker_type', type=str, default="fix_prefix_injection")
    parser.add_argument('--n_devices', type=int, default=2)
    parser.add_argument('--num_attacks_evaluated', type=int, default=30)
    args = parser.parse_args()

    main_test_set(args)

    # main_multiple_models("gpt-3.5-turbo-0613")

    # "model_name": "lmsys/vicuna-7b-v1.5",
    # "model_name": "allenai/OLMo-7B-Instruct",
    # "model_name": "/net/nfs.cirrascale/mosaic/kavelr/model_cache/llama-2-7b-chat-hf",
    # "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    # "model_name": "allenai/tulu-2-dpo-7b",
    # "model_name": "allenai/tulu-2-7b",
    # "model_name": "gpt-3.5-turbo-0613",
    # "model_name": "gpt-3.5-turbo-1106",
    # "model_name": "gpt-4-1106-preview",
    # "model_name": "gpt-4-0613",
    # "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "model_name": "/net/nfs.cirrascale/mosaic/liweij/auto_jailbreak/src/open-instruct-safety/models/mixture_v1.1.1",
    # "model_name": "/net/nfs.cirrascale/mosaic/oe-safety-models/tulu-2-7b-refusalv2-benign-v1-contrast-v1-mixture/"
