import os
import sys
import ray
import argparse
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

pd.set_option('display.max_columns', 500)

sys.path.append(os.getcwd())
from src.evaluation.eval_utils import *
from src.TacticsAnalyzer import TacticsAnalyzer
from src.configs.default import *
from src.my_utils import _init_ray
from src.ExptManager import ExptManager


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
            "attacker_config": attacker_config,
            "wandb_project_name": "wildteaming-diversity"}


def parse_args():
    parser = argparse.ArgumentParser(description='Diversity evaluation')
    parser.add_argument('--model_name', type=str, default="lmsys/vicuna-7b-v1.5", help='defender model')
    parser.add_argument('--num_attacks', type=int, default=10)
    parser.add_argument('--num_attacks_evaluated', type=int, default=10)
    parser.add_argument('--num_tactics_per_attack', type=int, default=3)
    parser.add_argument('--attacker_type', type=str,
                        default="fix_lead_seed_sentence")
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--unique_threshold', type=float, default=0.75)
    args = parser.parse_args()
    return args


def main(args):
    # need to use ray to manage loading multiple vllm models on the same machine
    expt_manager_config = get_expt_manager_config()
    _init_ray()

    ####### Set Custom Configs ########
    N = args.N
    num_attacks_evaluated = args.num_attacks_evaluated
    unique_threshold = args.unique_threshold
    expt_manager_config["defender_config"]["model_name"] = args.model_name
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

    tactics_save_dir = os.path.dirname(
        expt_manager.model_completion_filename) + "/tactics/"

    test_evaluator = get_evaluator("cais/HarmBench-Llama-2-13b-cls")
    tactics_analyzer = TacticsAnalyzer()
    embedder_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    all_behaviors = list(test_cases.keys())
    all_completions = expt_manager.load_model_completions()
    all_attacks = expt_manager.load_attacks()

    all_ASR_xN = {i: [] for i in range(1, N + 1)}
    all_Query_xN = {i: [] for i in range(1, N + 1)}
    all_sim_xN = {i: [] for i in range(1, N + 1)}
    selected_attacks = []
    selected_behaviors = []
    all_record = {}
    for behavior in tqdm(all_behaviors):
        print("=" * 100, f"\nBehavior: {behavior}\n", "=" * 100)

        behavior_attacks = all_attacks[behavior]["attacks"]
        behavior_attacks_prune_labels = all_attacks[behavior]["prune_labels"]
        behavior_attacks = [behavior_attacks[i] for i in range(len(behavior_attacks)) if behavior_attacks_prune_labels[i] == 0][:num_attacks_evaluated]
        
        behavior_completions = all_completions[behavior][:num_attacks_evaluated]
        behavior_attacks_test_labels = test_evaluator.evaluate(behavior, behavior_completions, behavior_attacks)[0]
        
        print("Num of model completions to evaluate:", len(behavior_attacks),
              len(behavior_completions),
              len(behavior_attacks_test_labels))


        # collecting the first successful attack from all behaviors for computing global metrics
        for attack, attack_test_label in zip(behavior_attacks, behavior_attacks_test_labels):
            if attack_test_label == 1:
                selected_attacks.append(attack)
                selected_behaviors.append(behavior)
                break

        all_record[behavior] = {
            "behavior": behavior,
            "behavior_attacks": behavior_attacks,
            "behavior_completions": behavior_completions,
            "behavior_attacks_test_labels": behavior_attacks_test_labels,
        }

        all_ASR_xN, all_Query_xN, all_sim_xN = main_behavior_diversity_metrics(N,
                                                                               embedder_model,
                                                                               behavior_attacks,
                                                                               behavior_attacks_test_labels,
                                                                               all_ASR_xN,
                                                                               all_Query_xN,
                                                                               all_sim_xN,
                                                                               unique_threshold)

        results_record = main_all_behavior_diversity_metrics(all_ASR_xN, all_Query_xN, all_sim_xN, N)

    ray.shutdown()

    flattened_results_record = flatten_diversity_results_record(results_record)
    expt_manager.log_results(None, flattened_results_record)

    results_record = main_global_diversity_metrics(embedder_model,
                                                   selected_attacks,
                                                   selected_behaviors,
                                                   tactics_analyzer,
                                                   tactics_save_dir,
                                                   results_save_path=None,
                                                   results_record=results_record)

    flattened_results_record = flatten_diversity_results_record(results_record)
    expt_manager.log_results(None, flattened_results_record)


if __name__ == '__main__':
    args = parse_args()
    main(args)
