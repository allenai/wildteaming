import time
from typing import List
import os
import ray
from tqdm import tqdm
from vllm import LLM, SamplingParams
import openai
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def is_initialized(self):
        print(f"==> Initialized {self.model_name}")

    def get_attribute(self, attr_name):
        return getattr(self, attr_name, None)


class GPT():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(self, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _generate(self,
                  prompt: str,
                  max_new_tokens: int,
                  temperature: float,
                  top_p: float,
                  system_message: str = None,
                  is_print_example: bool = False,
                  **kwargs):
        for _ in range(self.API_MAX_RETRY):
            try:
                if system_message is not None:
                    formatted_msg = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
                else:
                    formatted_msg = [{"role": "user", "content": prompt}]

                if is_print_example:
                    print("=" * 30, "An Example of formatted prompt", "=" * 30)
                    print(formatted_msg)
                    print("=" * 92)

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_msg,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except Exception as e:
                print("ERROR:", e, type(e))
                time.sleep(self.API_RETRY_SLEEP)
                if type(e) == openai.BadRequestError:
                    return "I'm sorry but the prompt is invalid. Please try again."
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def generate(self,
                 prompt: str,
                 max_new_tokens: int,
                 temperature: float,
                 top_p: float):
        return self._generate(prompt, max_new_tokens, temperature, top_p)

    def batched_generate(self,
                         prompts: List[str],
                         max_new_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,
                         use_tqdm: bool = False,
                         **kwargs):
        if use_tqdm:
            prompts = tqdm(prompts)
        return [self._generate(prompt, max_new_tokens, temperature, top_p, **kwargs) for prompt in prompts]


@ray.remote
class VLLM:
    def __init__(self, model_name_or_path, n_devices=1, **model_kwargs):
        self.model_name = model_name_or_path

        # In https://github.com/vllm-project/vllm/blob/main/vllm/engine/ray_utils.py
        # VLLM will manually allocate gpu placement_groups for ray actors if using tensor_parallel (num_gpus) > 1
        # So we will set CUDA_VISIBLE_DEVICES to all and let vllm.engine.ray_utils handle this
        # if n_devices > 1:
        resources = ray.cluster_resources()
        available_devices = ",".join([str(i) for i in range(int(resources.get("GPU", 0)))])  # "2,3,4,5"
        os.environ['CUDA_VISIBLE_DEVICES'] = available_devices

        self.model = self.load_vllm_model(model_name_or_path, num_gpus=n_devices, **model_kwargs)

    def batched_generate(self,
                         prompts: list[str],
                         do_chat_formatting: bool = False,
                         system_message: str = None,
                         tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
                         use_tqdm: bool = False,
                         return_full_outputs: bool = False,  # Whether to return the direct vllm output objects
                         temperature: float = 1.0,
                         top_p: float = 1.0,
                         max_tokens: int = 2048,
                         is_print_example: bool = False,
                         **sampling_args
                         ):
        if do_chat_formatting:
            assert tokenizer is not None, "Chat formatting requires tokenizer"
            if system_message is not None:
                conversation_prompts = [[{'role': 'system', 'content': system_message}, {'role': 'user', 'content': p}]
                                        for p in prompts]
            else:
                conversation_prompts = [[{'role': 'user', 'content': p}] for p in prompts]
            formatted_prompts = [tokenizer.apply_chat_template(p, tokenize=False) for p in conversation_prompts]
        else:
            formatted_prompts = prompts

        if is_print_example:
            print("=" * 30, "An Example of formatted prompt", "=" * 30)
            print(formatted_prompts[0])
            print("=" * 92)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **sampling_args
        )
        outputs = self.model.generate(prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        if return_full_outputs:
            return outputs

        results = [it.outputs[0].text for it in outputs]
        return results

    def load_vllm_model(self,
                        model_name_or_path,
                        dtype='auto',
                        trust_remote_code=True,  # False
                        download_dir=None,
                        revision=None,
                        quantization=None,
                        num_gpus=1,
                        ## tokenizer_args
                        use_fast_tokenizer=True,
                        pad_token=None,
                        eos_token=None,
                        **kwargs
                        ):

        model = LLM(model=model_name_or_path,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    download_dir=download_dir,
                    revision=revision,
                    quantization=quantization,
                    tokenizer_mode="auto" if use_fast_tokenizer else "slow",
                    tensor_parallel_size=num_gpus)

        if pad_token:
            model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
        if eos_token:
            model.llm_engine.tokenizer.tokenizer.eos_token = eos_token

        return model

    def set_tokenizer_truncation_side(self, side):
        self.model.llm_engine.tokenizer.tokenizer.truncation_side = side

    def is_initialized(self):
        print(f"==> Initialized {self.model_name}")
