base_prompts = {
    "lmsys/vicuna-7b-v1.5": "A chat between a curious human and an artificial intelligence assistant. "
                            "The assistant gives helpful, detailed, and polite answers to the human's questions. ",
}

system_prompts = {
    "lmsys/vicuna-7b-v1.5": {
        "original": base_prompts["lmsys/vicuna-7b-v1.5"] + "USER: {instruction} ASSISTANT:",
    },

    "llama-2-7b-chat-hf": {
        "original": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    },

    "mistralai/Mistral-7B-Instruct-v0.2": {
        "original": "[INST] {instruction} [/INST]",
    },

    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "original": "[INST] {instruction} [/INST]",
    },

    "gpt-3.5-turbo-0613": {
        "original": "You are a helpful assistant.",
    },

    "gpt-3.5-turbo-1106": {
        "original": "You are a helpful assistant.",
    },

    "gpt-4-1106-preview": {
        "original": "You are a helpful assistant.",
    },

    "gpt-4-0613": {
        "original": "You are a helpful assistant.",
    },

    "gpt-3.5-turbo": {
        "original": "You are a helpful assistant.",
    },

    "gpt-3.5-turbo-0125": {
        "original": "You are a helpful assistant.",
    },

    "gpt-4": {
        "original": "You are a helpful assistant.",
    },
    "open-instruct": {
        "original": "<|user|>\n{instruction}\n<|assistant|>\n",
    }
}
