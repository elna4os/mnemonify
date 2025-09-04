---
title: Mnemonify
emoji: 🇯🇵🈯️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: true
license: mit
short_description: Kanji Mnemonic Generator
---

# Mnemonify

- **Mnemonic** is a short story that helps to remember something. Mnemonify is a tool that generates mnemonics for Japanese kanji characters using a fine-tuned Phi-3-Mini-4K-Instruct model.
- You can try it on [HuggingFace Space](https://huggingface.co/spaces/elna4os/mnemonify) (**Caution**! Model has been deployed on free-tier instance, so the inference is painfully slow)
- Fine-tuned model [card](https://huggingface.co/elna4os/mnemonify)

---

<ins>Repository contains code for</ins>:

- [Fetching](scripts/fetch_wanikani.py) data from WaniKani API (you need to set up your own API key in .env file, see `.env_sample`)
- [Preparing](scripts/prepare_prompts.py) prompts for Phi-3-Mini-4K-Instruct fine-tuning
- [Fine-tuning](scripts/train_phi3_mini_4k.py) Phi-3-Mini-4K-Instruct with QLoRA
- [Merging/unloading](scripts/merge_unload.py) pretrained model
- Running an [app](app.py) with Streamlit

---

<ins>Notes</ins>:

- Python (>=3.11, <=3.13)
- At the time of writing, **bitsandbytes** is not compatible with M1+ processors (in development), thus fine-tuning is only available on x86_64 CPUs
- I used **RTX 4090 24GB RAM** for fine-tuning model with **QLoRA**
- Each stage params are described in [params.yaml](params.yaml) file

---

<ins>Data sources</ins>:

- 2083 kanji and 499 radicals from [WaniKani API](https://docs.api.wanikani.com/20170710/#introduction) are used to fine-tune the model and make inference. Remember that WaniKani updates their data (subjects) from time to time, so the actual number of kanji/radicals and their meainings/readings/mnemonics/etc can be different
- [KanjiAlive](https://github.com/kanjialive/kanji-data-media) radicals/kanji and [KRAD files](https://www.edrdg.org/krad/kradinf.html) (kanji decomposition) are used to prepare prompts during "fallback" inference mode

_Fallback mode_ - when the kanji is not found in WaniKani subjects, the app uses KanjiAlive and KRAD data to prepare prompts (if possible)

Remember that "radical" is a very loosely defined term in Japanese language, so decomposing kanji into radicals is not always straightforward and consistent between different sources (WaniKani, KRAD, etc.)

---

<ins>To do</ins>:

- Find a reliable and open-source way to decompose kanji into radicals (unfortunately, WaniKani license doesn't allow to publicly share their decompositions with according meanings). KRAD decomposition is too atomic which apparently is not really good for mnemonics and often not consistent with WaniKani mnemonics. This strongly affects final user experience, because current mnemonics are generated without references to compounding radicals and their meanings
- I didn't experimented much with hyperparameters, prompt templates and so on, so there is definitely room for improvement
- Fine-tune the model to generate mnemonics for vocabulary (sequence of 2 or more kanji characters). Luckily, this one could be done very easily, because the kanji meaning is not a secret
