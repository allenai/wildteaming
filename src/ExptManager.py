import os
import sys
import wandb
import os.path
import pandas as pd

sys.path.append(os.getcwd())
from src.utils import *
from src.Defender import *

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
            data_path = f"src/data/harmbench/harmbench_behaviors_text_{split}.csv"
            df_data = pd.read_csv(data_path)
            if self.expt_config["data_types"] != "all":
                data_types = self.expt_config["data_types"].split("_")
                df_data = df_data[df_data["FunctionalCategory"].isin(data_types)]
            all_data = df_data.to_dict(orient='records')
            self.test_cases = {d["Behavior"]: {"behavior_info": d} for d in all_data}

            # load target string map
            data_path = "src/data/harmbench/harmbench_targets_text.json"
            with open(data_path, "r") as f:
                self.behavior_target_map = json.load(f)

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
                        self.final_evaluator_filename]
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
