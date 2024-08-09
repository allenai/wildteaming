"""
This file is the main file for the Attacker class.
The Attacker class is used to generate attacks for a given behavior.
"""
import os
import sys
import ray
from transformers import AutoTokenizer

sys.path.append(os.getcwd())
from src.language_models import *
from data.prompts.system_prompts import system_prompts
from src.my_utils import *


class Attacker:
    def __init__(self, attacker_config, model=None, tokenizer=None):
        self.COMMON_TACTICS_THRESHOLD = 2
        self.COMMON_TACTICS_TOP_K = 500
        self.attacker_config = attacker_config
        self.tokenizer = None

        self._load_tactics_map()
        self._load_compose_tactics_prompt()

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model is not None and tokenizer is None:
            self.model = model
        elif model is None and tokenizer is not None:
            raise ValueError("Model not provided")
        else:
            self._init_attacker()

    def reload_config_shallow(self, attacker_config):
        """
        Reloads the attacker config without reloading the already loaded attack model.
        """
        self.attacker_config = attacker_config

    def _init_attacker(self):
        print("Initializing attacker model...")
        if "mistralai" in self.attacker_config["model_name"]:
            self._int_vllm_attacker()
        elif "gpt" in self.attacker_config["model_name"]:
            self.model = GPT(self.attacker_config["model_name"])
        else:
            raise ValueError("Model not supported")

    def _int_vllm_attacker(self):
        """
        Initializes the VLLM model for the attacker.
        """
        if self.attacker_config["is_vllm"]:
            self.model = VLLM.remote(self.attacker_config["model_name"], **self.attacker_config)
            ray.get(self.model.is_initialized.remote())  # init ray model
            self.tokenizer = AutoTokenizer.from_pretrained(self.attacker_config["model_name"])
        else:
            raise ValueError("Non-vllm model not supported")

    def _load_tactics_map(self):
        """
        Loads the tactics map amd the frequency map.
        """
        data_path = "data/tactics/auto_tactics.json"
        with open(data_path, "r") as f:
            self.all_tactics_map = json.load(f)
        self.tactic_frequency_map = read_json("data/tactics/auto_tactics_frequency.json")

        # ignore some ill-formed tactics and tactics related to explicit content
        to_exclude_tactics = ["strategy name", "name of the first new strategy", ""]
        does_not_contain = ["explicit", "sex", "erotic"]
        self.all_tactics_map = {k: v for k, v in self.all_tactics_map.items()
                                if (k not in to_exclude_tactics)
                                and k[-1] != ":"
                                and (all(x not in v["definition"][0].lower() for x in does_not_contain))
                                and (all(x not in k.lower() for x in does_not_contain))}

        self.tactic_frequency_map = {k: v for k, v in self.tactic_frequency_map.items() if k in self.all_tactics_map}
        self.tactic_frequency_list = [(k, v["count"]) for k, v in self.tactic_frequency_map.items()]
        print("Loaded tactics map from:", data_path)

    def _load_compose_tactics_prompt_file(self, filename):
        with open(filename, 'r') as f:
            self.base_prompt = f.read()

    def _load_compose_tactics_prompt(self):
        """
        Loads the prompt for composing jailbreak tactics.
        """
        prompt_base_path = "data/prompts"
        if self.attacker_config["attacker_type"] == "original":
            self._load_compose_tactics_prompt_file(
                f"{prompt_base_path}/compose_tactics.txt")

        elif self.attacker_config["attacker_type"] == "without_lead_seed_sentence_example":
            self._load_compose_tactics_prompt_file(
                f"{prompt_base_path}/compose_tactics_without_lead_seed_sentence_example.txt")

        elif self.attacker_config["attacker_type"] == "fix_lead_seed_sentence":
            self._load_compose_tactics_prompt_file(
                f"{prompt_base_path}/compose_tactics_fix_lead_seed_sentence.txt")

        else:
            raise ValueError("Attacker type not supported")

    def _convert_tactics_list_to_string(self, tactics, tactics_definition, tactics_examples):
        """
        Converts the tactics list to a formatted string.
        """
        tactics_string = ""
        for t, d, es in zip(tactics, tactics_definition, tactics_examples):
            tactics_string += f"- {t}: [definition] {d} "
            tactics_string += " [example] ".join(es)
            tactics_string += "\n"
        return tactics_string

    def _format_attack_single(self, behavior, tactics, target_str=None):
        """
        Format the input string for a single attack.
        """
        tactics_definition = []
        tactics_examples = []
        for t in tactics:
            tactics_definition.extend(random.sample(self.all_tactics_map[t]["definition"], 1))
            tactics_examples.append(random.sample(self.all_tactics_map[t]["excerpt"],
                                                  self.attacker_config["num_excerpts_per_tactic"]))
        formatted_strategies_list = self._convert_tactics_list_to_string(tactics, tactics_definition, tactics_examples)

        compose_prompt = self.base_prompt.replace("[INSERT STRATEGY LIST]", formatted_strategies_list)
        compose_prompt = compose_prompt.replace("[INSERT SIMPLE PROMPT]", behavior)
        if target_str is not None:
            compose_prompt = compose_prompt.replace("[INSERT STARTING STRING]", target_str)

        return compose_prompt

    def _select_tactics(self):
        """
        Selects jailbreak tactics for generating the attack based on different selection methods.
        """

        # Randomly select tactics from the tactics map.
        if self.attacker_config["tactics_selection_method"] == "random":
            return random.sample(list(self.all_tactics_map.keys()), self.attacker_config["num_tactics_per_attack"])

        # Randomly select tactics from the tactics map, but prioritize common tactics.
        elif self.attacker_config["tactics_selection_method"] == "random_common_prioritized":
            all_common_tactics = [k for k, v in self.all_tactics_map.items()
                                  if len(v["excerpt"]) > self.COMMON_TACTICS_THRESHOLD]
            all_uncommon_tactics = [k for k, v in self.all_tactics_map.items()
                                    if len(v["excerpt"]) <= self.COMMON_TACTICS_THRESHOLD]

            num_uncommon_to_select = self.attacker_config["num_tactics_per_attack"] // 2
            num_common_to_select = self.attacker_config["num_tactics_per_attack"] - num_uncommon_to_select

            selected_common = random.sample(all_common_tactics, num_common_to_select)
            selected_uncommon = random.sample(all_uncommon_tactics, num_uncommon_to_select)
            return selected_common + selected_uncommon

        # Randomly select tactics from the most common tactics.
        elif self.attacker_config["tactics_selection_method"] == "random_among_most_common":
            most_common_tactics = [k for k, v in self.tactic_frequency_list[:self.COMMON_TACTICS_TOP_K]]

            selected_tactics = []
            selected_tactic_types = []
            while len(selected_tactics) < self.attacker_config["num_tactics_per_attack"]:
                tt = random.choice(most_common_tactics)
                if tt not in selected_tactic_types:
                    selected_tactic_types.append(tt)
                    # sample a tactic name under the selected tactic type
                    t = random.choice(self.tactic_frequency_map[tt]["names"])
                    if t not in self.all_tactics_map:
                        selected_tactics.append(tt)
                    else:
                        selected_tactics.append(t)
            return selected_tactics

    def _format_all_attacks_gen_prompts(self, behavior, num_attacks, target_str):
        """
        Format the input strings for all attacks.
        """
        # select jailbreak tactics to use for all attacks
        all_tactics = [self._select_tactics() for _ in range(num_attacks)]
        all_attacks_gen_prompts = [
            self._format_attack_single(behavior, attack_tactics, target_str)
            for attack_tactics in all_tactics
        ]
        return all_attacks_gen_prompts, all_tactics

    def _parse_attacks(self, all_attacks):
        """
        Parse the attacks generated by the model to remove formatting errors.
        """
        parsed_attacks = []
        for a in all_attacks:
            a = a.replace("---\n", "\n")
            a_split = a.split("\n")

            # only keeping the content after the "adversarial prompt:" line
            if "adversarial prompt" in a.lower() and ":" in a:
                for i, s_a in enumerate(a_split):
                    if "adversarial prompt" in s_a.lower() and ":" in s_a and i < len(a_split) - 1:
                        a_split = a_split[i + 1:]
                        break

            # stripping out the "revision strategies" section
            revision_bullets = ["-", "*"]
            if "revision strategies" in a.lower() or " prompt:" in a.lower():
                for i, s_a in enumerate(a_split):
                    if (("revision strategies" in s_a.lower() and ":" in s_a) or
                            ("simple prompt" in s_a.lower() and ":" in s_a)):
                        a_split.remove(s_a)
                        for j, s_a in enumerate(a_split[i:]):
                            if any(a.startswith(b) for b in revision_bullets):
                                a_split.remove(s_a)
                            elif any(char.isdigit() for char in s_a) and ". " in s_a:
                                a_split.remove(s_a)
                        break

            # removing line with "adversarial" or "revision strategies" in it
            last_line_keywords = ["adversarial", "revision strategies", "confidence:"]
            if any(kw in a.lower() for kw in last_line_keywords):
                for s_a in a_split:
                    if any(kw in s_a.lower() for kw in last_line_keywords):
                        a_split.remove(s_a)

            a_r = "\n".join(a_split)
            a_r = a_r.strip("\n")
            a_r = a_r.strip("-")
            a = a_r.strip(" ")
            if len(a) >= 2 and a.startswith("\"") and a.endswith("\""):
                a = a[1:-1]
            parsed_attacks.append(a)
        return parsed_attacks

    def get_attacks(self, behavior, num_attacks, target_str="Sure, here's ..."):
        """
        Public method to get attacks.
        :param behavior: the harmful behavior to generate attacks for.
        :param num_attacks: the number of attacks to generate for the given behavior.
        :param target_str (Optional): the target string aimed by the attack, for the prefix-appending tactic.
        :return:
        """
        all_attacks_gen_prompts, all_tactics = self._format_all_attacks_gen_prompts(behavior, num_attacks, target_str)

        if "mistralai" in self.attacker_config["model_name"]:
            return self._get_attacks_mistral(all_attacks_gen_prompts, all_tactics)
        elif "gpt" in self.attacker_config["model_name"]:
            return self._get_attacks_gpt(all_attacks_gen_prompts, all_tactics)

    def _get_attacks_gpt(self, all_attacks_gen_prompts, all_tactics):
        """
        Get attacks using the GPT model.
        """
        all_raw_attacks = self.model.batched_generate(all_attacks_gen_prompts,
                                                      max_new_tokens=self.attacker_config["num_tokens"],
                                                      temperature=self.attacker_config["temperature"],
                                                      top_p=self.attacker_config["top_p"],
                                                      use_tqdm=True,
                                                      system_message=system_prompts[self.attacker_config["model_name"]]
                                                      ["original"])
        all_attacks = self._parse_attacks(all_raw_attacks)
        return all_raw_attacks, all_attacks, all_tactics

    def _get_attacks_mistral(self, all_attacks_gen_prompts, all_tactics):
        """
        Get attacks using the Mistral model.
        """
        all_attacks = self.model.batched_generate.remote(prompts=all_attacks_gen_prompts,
                                                         do_chat_formatting=True,
                                                         tokenizer=self.tokenizer,
                                                         use_tqdm=True,
                                                         return_full_outputs=False,
                                                         temperature=self.attacker_config["temperature"],
                                                         top_p=self.attacker_config["top_p"],
                                                         max_tokens=self.attacker_config["num_tokens"])
        all_raw_attacks = ray.get(all_attacks)
        all_attacks = self._parse_attacks(all_raw_attacks)
        return all_raw_attacks, all_attacks, all_tactics
