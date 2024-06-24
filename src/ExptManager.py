import os
import sys
import wandb
import os.path
import pandas as pd

sys.path.append(os.getcwd())
from src.utils.main_utils import *
from src.jailbreak_baselines.wildteaming.Pruner import *
from src.jailbreak_baselines.wildteaming.Defender import *

random.seed(42)


class ExptManager:
    def __init__(self, expt_config):
        self.expt_config = expt_config
        self._init_test_cases()
        self._init_save_dir()
        if expt_config["is_wandb"]:
            self._init_logger(self.expt_config["wandb_project_name"])

    def _init_logger(self, project_name):
        """
        Initialize the wandb logger.
        """
        num_tactics_per_attack = self.expt_config["attacker_config"]["num_tactics_per_attack"]
        attacker_type = self.expt_config["attacker_config"]["attacker_type"]

        self.logger = wandb.init(project=project_name,
                                 name=f"{attacker_type}={num_tactics_per_attack}",
                                 config=self.expt_config)

    def _init_test_cases(self):
        """
        Load test cases.
        """
        if self.expt_config["dataset"] == "harmbench":
            split = self.expt_config["split"]
            data_path = f"src/HarmBench/data/behavior_datasets/harmbench_behaviors_text_{split}.csv"
            df_data = pd.read_csv(data_path)
            if self.expt_config["data_types"] != "all":
                data_types = self.expt_config["data_types"].split("_")
                df_data = df_data[df_data["FunctionalCategory"].isin(data_types)]
            all_data = df_data.to_dict(orient='records')
            self.test_cases = {d["Behavior"]: {"behavior_info": d} for d in all_data}

            # load target string map
            data_path = "src/HarmBench/data/optimizer_targets/harmbench_targets_text.json"
            with open(data_path, "r") as f:
                self.behavior_target_map = json.load(f)

        elif self.expt_config["dataset"] == "merged_and_shuffled_vanilla.v.02_train":
            data_path = "/net/nfs.cirrascale/mosaic/oe-safety-datasets/v0_2/categories/seungju_new_completions/tulu_format/merged_and_shuffled_vanilla.v.02_train.jsonl"
            with open(data_path, 'r') as f:
                all_data = [json.loads(l) for l in f]
            self.test_cases = {d["messages"][0]["content"]: d for d in all_data}

        elif self.expt_config["dataset"] == "xstest_expanded/v1":
            data_path = "/net/nfs.cirrascale/mosaic/liweij/auto_jailbreak/data/safety_training_data/xstest_expanded/v1.jsonl"
            all_data = load_standard_data(data_path)
            self.test_cases = {d["messages"][0]["content"]: d for d in all_data}

        elif self.expt_config["dataset"] == "unused_vani_h_prompts_dedup_minhash":
            data_path = "/net/nfs.cirrascale/mosaic/liweij/auto_jailbreak/data/safety_training_data/v3/raw/unused_vani_h_prompts_dedup_minhash.jsonl"
            all_data = load_standard_data(data_path)
            self.test_cases = {d["prompt"]: d for d in all_data}

        elif self.expt_config["dataset"] == "xstest_expanded/v2":
            data_path = "/net/nfs.cirrascale/mosaic/oe-safety-datasets/contrastive_benign_vanilla_data/gpt-4_responses_harmful_prompts_diverse_v2.json"
            all_data = load_standard_data(data_path)[0]["subcategories"][0]["fine_grained_subcategories"]
            self.test_cases = {}
            for d in all_data:
                for e in d["examples"]:
                    self.test_cases[e] = {}

        elif self.expt_config["dataset"] == "ai2_v0_3_harmful_prompts_train":
            data_path = "/net/nfs.cirrascale/mosaic/oe-safety-datasets/202405_wildguard/ai2_v0_3_harmful_prompts_train.jsonl"
            all_data = load_standard_data(data_path)
            self.test_cases = {}
            for d in all_data:
                p = d["prompt"]
                self.test_cases[p] = {}
            print("Num Test Cases:", len(self.test_cases))

        else:
            raise NotImplementedError

    def get_test_cases(self):
        """
        Get loaded test cases.
        """
        return self.test_cases

    def get_behavior_target_map(self):
        """
        Get behavior target map.
        """
        return self.behavior_target_map

    def reset_config(self, expt_config):
        """
        Reset the experiment configuration.
        """
        self.expt_config = expt_config
        self._init_test_cases()
        self._init_save_dir()

    def _init_save_dir(self):
        """
        Initialize the save directories for the experiment.
        """
        config_to_exclude_in_path = ["model_name", "is_vllm", "n_devices",
                                     "do_chat_formatting", "use_tqdm", "trust_remote_code",
                                     "is_sequential"]
        dataset = self.expt_config["dataset"]
        split = self.expt_config["split"]
        data_types = self.expt_config["data_types"]
        defender_config = self.expt_config["defender_config"]
        attacker_config = self.expt_config["attacker_config"]
        final_evaluator_config = self.expt_config["final_evaluator_config"]
        intermediate_evaluators_config = self.expt_config["intermediate_evaluators_config"]

        ######## attack save path ########
        attacker_model_name = attacker_config["model_name"].replace("/", "--")
        attacker_config_include = {k: v for k, v in attacker_config.items() if k not in config_to_exclude_in_path}
        attacker_config_str = "=".join([f"{k}={v}" for k, v in attacker_config_include.items()]).replace("/", "--")

        attack_base_path = os.path.join(self.expt_config["base_save_path"]
                                        + f"{dataset}/{split}/{data_types}/"
                                        + f"attacker/"
                                          f"{attacker_model_name}/{attacker_config_str}/")
        self.attack_filename = attack_base_path + "attacks.json"

        ######## defender model completions save path ########
        defender_model_name = defender_config["model_name"].replace("/", "--")
        defender_config_include = {k: v for k, v in defender_config.items() if k not in config_to_exclude_in_path}
        defender_config_str = "=".join([f"{k}={v}" for k, v in defender_config_include.items()]).replace("/", "--")
        model_completion_base_path = (attack_base_path
                                      + f"defender/{defender_model_name}/"
                                      + f"{defender_config_str}/")
        self.model_completion_filename = model_completion_base_path + "model_completions.json"

        ######## intermediate evaluators save path ########
        self.intermediate_evaluators_filenames = []
        for eval_config in intermediate_evaluators_config:
            eval_model_name = eval_config["model_name"].replace("/", "--")
            eval_config_include = {k: v for k, v in eval_config.items() if k not in config_to_exclude_in_path}
            eval_config_str = "=".join([f"{k}={v}" for k, v in eval_config_include.items()]).replace("/", "--")
            intermediate_evaluators_base_path = (model_completion_base_path
                                                 + f"intermediate_evaluators/model_name={eval_model_name}="
                                                 + f"{eval_config_str}/")
            eval_filename = intermediate_evaluators_base_path + "intermediate_evaluations.json"
            self.intermediate_evaluators_filenames.append(eval_filename)

        ######## final evaluator save path ########
        final_evaluator_model_name = final_evaluator_config["model_name"].replace("/", "--")
        final_evaluator_config_include = {k: v for k, v in final_evaluator_config.items()
                                          if k not in config_to_exclude_in_path}
        final_evaluator_config_str = "=".join([f"{k}={v}" for k, v in final_evaluator_config_include.items()]).replace(
            "/", "--")
        final_evaluator_base_path = (model_completion_base_path
                                     + f"final_evaluator/model_name={final_evaluator_model_name}="
                                     + f"{final_evaluator_config_str}/")
        self.final_evaluator_filename = final_evaluator_base_path + "final_evaluation.json"

        all_filanems = [self.attack_filename, self.model_completion_filename,
                        self.final_evaluator_filename] + self.intermediate_evaluators_filenames
        for p in all_filanems:
            if not os.path.exists(os.path.dirname(p)):
                os.makedirs(os.path.dirname(p))
                print("- Created path:", os.path.dirname(p))

        ######## save attack config ########
        config_to_save_path = attack_base_path + "config.json"
        if not os.path.exists(config_to_save_path):
            with open(config_to_save_path, 'w') as f:
                json.dump(attacker_config, f)
            print("=" * 20, f"\nAttack config saved.")

        ######## save defender model completions config ########
        config_to_save_path = model_completion_base_path + "config.json"
        if not os.path.exists(config_to_save_path):
            with open(config_to_save_path, 'w') as f:
                json.dump(defender_config, f)
            print("=" * 20, f"\nDefender model completions config saved.")

        ######## save intermediate evaluators config ########
        for eval_config, eval_filename in zip(intermediate_evaluators_config, self.intermediate_evaluators_filenames):
            config_to_save_path = os.path.dirname(eval_filename) + "/config.json"
            if not os.path.exists(config_to_save_path):
                with open(config_to_save_path, 'w') as f:
                    json.dump(eval_config, f)
                print("=" * 20, f"\nIntermediate evaluator config saved.")

        ######## save final evaluator config ########
        config_to_save_path = final_evaluator_base_path + "config.json"
        if not os.path.exists(config_to_save_path):
            with open(config_to_save_path, 'w') as f:
                json.dump(final_evaluator_config, f)
            print("=" * 20, f"\nFinal evaluator config saved.")

    def save_attacks(self, attacks):
        """
        Save attacks to file.
        """
        with open(self.attack_filename, 'w') as f:
            json.dump(attacks, f)
        print("=" * 20, f"\nAttacks saved to ...")
        print(self.attack_filename)
        print("=" * 20)

    def save_model_completions(self, model_completions):
        """
        Save model completions to file.
        """
        with open(self.model_completion_filename, 'w') as f:
            json.dump(model_completions, f)
        print("=" * 20, f"\nModel completions saved to ...")
        print(self.model_completion_filename)
        print("=" * 20)

    # def save_intermediate_evaluations(self, intermediate_evaluator_idx, intermediate_evaluations):
    #     with open(self.intermediate_evaluators_filenames[intermediate_evaluator_idx], 'w') as f:
    #         json.dump(intermediate_evaluations, f)
    #     print("=" * 20, f"\nIntermediate evaluations saved to ...")
    #     print(self.intermediate_evaluators_filenames[intermediate_evaluator_idx])
    #     print("=" * 20)
    #
    # def save_final_evaluation(self, final_evaluation):
    #     with open(self.final_evaluator_filename, 'w') as f:
    #         json.dump(final_evaluation, f)
    #     print("=" * 20, f"\nFinal evaluation saved to ...")
    #     print(self.final_evaluator_filename)
    #     print("=" * 20)

    def load_attacks(self):
        """
        Load attacks from file.
        """
        if not os.path.exists(self.attack_filename):
            print("=" * 20, f"\nAttacks not found at ...")
            print(self.attack_filename)
            print("=" * 20)
            return None

        with open(self.attack_filename, 'r') as f:
            attacks = json.load(f)
        print("=" * 20, f"\nAttacks loaded from ...")
        print(self.attack_filename)
        print("=" * 20)
        return attacks

    def load_model_completions(self):
        """
        Load model completions from file.
        :return:
        """
        if not os.path.exists(self.model_completion_filename):
            print("=" * 20, f"\nModel completions not found at ...")
            print(self.model_completion_filename)
            print("=" * 20)
            return None

        with open(self.model_completion_filename, 'r') as f:
            model_completions = json.load(f)
        print("=" * 20, f"\nModel completions loaded from ...")
        print(self.model_completion_filename)
        print("=" * 20)
        return model_completions

    def log_results(self, logs, final_results):
        """
        Log the results to wandb.
        """
        if logs is not None:
            logs = [v for k, v in logs.items()]
            df_logs = pd.DataFrame(logs)
            final_results["final_evaluation"] = df_logs
        self.logger.log(final_results)

    def log_final_results(self,
                          final_evaluation,
                          all_scores=None,
                          avg_num_attacks_to_succeed=None,
                          avg_durations=None):
        final_evaluation_list = [v for k, v in final_evaluation.items()]
        df_final_evaluation = pd.DataFrame(final_evaluation_list)
        self.logger.log({
            "all_scores": all_scores,
            "avg_num_attacks_to_succeed": avg_num_attacks_to_succeed,
            "avg_durations": avg_durations,
            "final_evaluation": wandb.Table(dataframe=df_final_evaluation)})
