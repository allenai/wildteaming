# <img src=assets/wildteaming_logo.png width=40/> WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models

<p align="center">

[//]: # (  <a href="https://allenai.github.io/lumos/">)

[//]: # (    <img src="https://img.shields.io/badge/🌐-Website-red">)

[//]: # (  </a>)

  <a href="https://arxiv.org/abs/2406.18510">
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

We introduce <img src=assets/wildteaming_logo.png width=25/> [WildTeaming](https://arxiv.org/pdf/2406.18510), an automatic red-teaming framework that mines *in-the-wild* user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes selections of multiple mined tactics for systematic exploration of novel and even more challenging jailbreaks. WildTeaming intends to address two challenges: 
- 🔍 Broadly identifying jailbroken behaviors of LLMs.
- 🛠️ Creating a publicly open, large-scale safety training resource for systematic defense (WildJailbreak).

For more findings, please refer to our [paper](https://arxiv.org/abs/2406.18510)!

<hr>
<img src=assets/wildteaming.png width=900/>

## Resources

  - 📑 **Paper**: [arXiv](https://arxiv.org/abs/2406.18510)
  - 🤗 **Models**: [7B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) and [13B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) safety-trained Tulu2 models
  - 🤗 **Data**: [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) *training* and *evaluation* datasets

## Mine Jailbreak Tactics In-the-Wild

WildTeaming mines jailbreak tactics from in-the-wild user-chatbot interactions (i.e., LMSYS, WildChat), resulting in a more diverse repository of novel jailbreak tactics than previous resources.

<img src=assets/jailbreak_tactics.png width=700/>


## WildTeaming for Automatic Jailbreaking

<img src=assets/jailbreak_results_breakdown.png width=700/>

## WildJailbreak Dataset

With WildTeaming, we create [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak), a large-scale open-source synthetic safety dataset with 262K *vanilla* (direct request) and *adversarial* (complex jailbreak) prompt-response pairs. We identify the training properties that enable an ideal balance of safety behaviors: **appropriate safeguarding without over-refusal, effective handling of both vanilla and adversarial queries, and minimal, if any, decrease in general capabilities.** To achieve such balance, WildJailbreak offer the following four types of data:

- **Vanilla Harmful**: direct requests that could potentially elicit harmful responses from LMs.
- **Vanilla Benign**: harmless prompts used to combat exaggerated safety, i.e., over-refusal on benign queries.
- **Adversarial Harmful**: adversarial jailbreaks that convey harmful requests in more convoluted and stealthy ways.
- **Adversarial Benign**: adversarial queries that look like jailbreaks but contain no harmful intent. 

<img src=assets/data_example_four_types.png width=700/>

## Citation

If you find it helpful, please feel free to cite our work!
```
@misc{wildteaming2024,
      title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models}, 
      author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18510},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18510}, 
}
```

## 🔥 News
- **[2024, June 25]**
  - 📑 **Paper** We release the WildTeaming paper on arXiv!
  - 🤗 **Models** We release the [7B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) and [13B](https://huggingface.co/allenai/llama2-13b-WildJailbreak) safety-trained Tulu2 models on Huggingface!
  - 🤗 **Data** We release the [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) *training* and *evaluation* datasets on Huggingface! 

