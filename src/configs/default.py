def get_final_evaluator_default_config():
    return {"model_name": "cais/HarmBench-Llama-2-13b-cls",
            "evaluator_type": "harmbench",
            "num_tokens": 512,
            # vllm specific
            "n_devices": 2}


def get_defender_default_config():
    return {
        "model_name": "lmsys/vicuna-7b-v1.5",
        "model_mode": "original",
        "num_tokens": 512,
        "temperature": 0,
        "top_p": 1.0,
        "use_tqdm": True,
        # vllm specific
        "is_vllm": True,
        "n_devices": 2,
        "do_chat_formatting": False,
        "trust_remote_code": True,
    }


def get_attacker_default_config():
    return {"model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "attacker_type": "original",
            "num_tokens": 1024,
            "temperature": 1,
            "top_p": 0.9,
            "num_tactics_per_attack": 4,
            "num_excerpts_per_tactic": 0,
            "tactics_selection_method": "random_common_prioritized",
            "num_attacks": 60,
            # vllm specific
            "is_vllm": True,
            "n_devices": 4,
            }
