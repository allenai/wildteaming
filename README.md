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

