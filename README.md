# <img src=assets/wildteaming_logo.png width=40/> WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models
<p align="center">

[//]: # (  <a href="https://allenai.github.io/lumos/">)

[//]: # (    <img src="https://img.shields.io/badge/🌐-Website-red">)

[//]: # (  </a>)

  <a href="https://arxiv.org/abs/xx">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://huggingface.co/datasets/allenai/wildjailbreak">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>
  <a href="https://huggingface.co/allenai/llama2-7b-WildJailbreak">
    <img src="https://img.shields.io/badge/🤗-Model-green">
  </a>

[//]: # (  <a href="https://huggingface.co/spaces/ai2lumos/lumos_data_demo">)

[//]: # (    <img src="https://img.shields.io/badge/🤗-Demo-yellow">)

[//]: # (  </a>)
</p>

**Authors:**
[Liwei Jiang](https://liweijiang.me),
[Kavel Rao](https://kavelrao.dev) ⭐,
[Seungju Han](https://seungjuhan.me) ⭐,
[Allyson Ettinger](https://aetting.github.io),
[Faeze Brahman](https://fabrahman.github.io),
[Sachin Kumar](https://sites.google.com/view/sachinkumar),
[Niloofar Mireshghallah](https://homes.cs.washington.edu/~niloofar/),
[Ximing Lu](https://scholar.google.com/citations?user=ssYPSmkAAAAJ&hl=en),
[Maarten Sap](http://maartensap.com),
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),
[Nouha Dziri](https://nouhadziri.github.io/)
&nbsp; &nbsp; &nbsp; ⭐ Co-second authors

We introduce <img src=assets/wildteaming_logo.png width=25/> WildTeaming, an automatic red-teaming framework that mines *in-the-wild* user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes selections of multiple mined tactics for systematic exploration of novel and even more challenging jailbreaks.

<img src=assets/wildteaming.png width=900/>

## WildTeaming Highlights

We introduce the WildTeaming framework to address two challenges: 

- broadly identifying jailbroken behaviors of LLMs
- creating a publicly open, large-scale safety training resource for systematic defense.

Compared to prior work that performed red-teaming via recruited human workers, gradient-based optimization, or iterative revision with large language models (LLMs), our work investigates jailbreaks from chatbot users in-the-wild who were not specifically instructed to break the system. WildTeaming reveals previously unidentified vulnerabilities of frontier LLMs, resulting in up to 4.6x more *diverse* and *successful* adversarial attacks compared to state-of-the-art jailbreaking methods.


## WildJailbreak Dataset

With WildTeaming, we create WildJailbreak, a large-scale open-source synthetic safety dataset with 262K *vanilla* (direct request) and *adversarial* (complex jailbreak) prompt-response pairs. We identify the training properties that enable an ideal balance of safety behaviors: **appropriate safeguarding without over-refusal, effective handling of both vanilla and adversarial queries, and minimal, if any, decrease in general capabilities.** To achieve such balance WildJailbreak offer the following four types of prompt-response pairs:



- **Vanilla Harmful**: direct requests that could potentially elicit harmful responses from LMs. We apply GPT-4 to synthetically generate 50,050 vanilla harmful prompts across 13 risk categories, inspired by taxonomy from Weidinger et al. In addition, we pair the harmful prompts with helpful and detailed refusal responses, also synthetically generated with GPT-3.5.
- **Vanilla Benign**: harmless prompts used to combat exaggerated safety, i.e., over-refusal on benign queries. Motivated by the exaggerated safety categories in XSTest, we use GPT-4 to generate 50,050 prompts that superficially resemble unsafe prompts by keywords or discuss sensitive topics in non-harmful ways. Similarly, we use GPT-3.5 to generate complying responses.
- **Adversarial Harmful**: adversarial jailbreaks that convey harmful requests in more convoluted and stealthy ways. We apply WildTeaming to transform our vanilla harmful queries with 2-7 randomly sampled In-the-Wild jailbreak tactics, with both the Mixtral-8×7B and GPT-4 models. We also filter out low-risk or off-topic prompts, and pair the model refusal responses generated from the counterpart vanilla prompts to adversarial prompts, yielding 82,728 items in this split of the dataset.
- **Adversarial Benign**: adversarial queries that look like jailbreaks but contain no harmful intent. Similar to adversarial harmful queries, we create 78,706 adversarial benign queries using WildTeaming, based on the vanilla (B) prompts. We use GPT-3.5 to generate direct continuations of the prompts as the target model response.


## Models




## Citation

If you find it helpful, please feel free to cite our work!
```
@misc{wildteaming2024,
      title={{WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models}}, 
      author={{Liwei Jiang and Seungju Han and Kavel Rao and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Nouha Dziri and Yejin Choi}}
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 🔥 News
- **[2024, June 25]**
  - 📑 **Paper** We release the WildTeaming paper on arXiv!
  - 🤗 **Models** We release the [7B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) and [13B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) safety-trained Tulu2 models on Huggingface!
  - 🤗 **Data** We release the [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) *training* and *evaluation* datasets on Huggingface! 

