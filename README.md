---
title: Mnemonify
emoji: 🈶
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: true
license: mit
short_description: Kanji Mnemonic Generator
---

### Generate kanji mnemonics with fine-tuned Phi-3-Mini-4K-Instruct

You can try it on [HF Spaces](https://huggingface.co/spaces/elna4os/mnemonify) (**Caution**! Model has been deployed on free-tier instance (2 vCPU), so the inference is painfully slow 🙂)

---

<ins>Requirements</ins>:

- Python (>=3.11, <=3.13)
- At the time of writing, **bitsandbytes** is not compatible with M1+ processors (in development), thus fine-tuning is only available on x86_64 CPUs.
- I used **RTX 4090 24GB RAM** for fine-tuning model with **QLoRA**

---

<ins>Repository contains code for</ins>:

- [Fetching](scripts/fetch_wanikani.py) data from WaniKani API (you need to set up your own API key in .env file, see `.env_sample`)
- [Preparing](scripts/prepare_prompts.py) prompts for Phi-3-Mini-4K-Instruct fine-tuning
- [Fine-tuning](scripts/train_phi3_mini_4k.py) Phi-3-Mini-4K-Instruct with QLoRA
- [Merging/unloading](scripts/merge_unload.py) pretrained model
- [Running](app.py) Streamlit app with fine-tuned model

---

<ins>Data sources and their usage</ins>:

- 2083 kanji from [WaniKani API](https://docs.api.wanikani.com/20170710/#introduction) has been used to fine-tune the model and to prepare prompts during inference stage.
- [KanjiAlive](https://github.com/kanjialive/kanji-data-media) radicals and kanji are used to prepare prompts during inference stage in fallback mode.
- [KRAD files](https://www.edrdg.org/krad/kradinf.html) are used to decompose kanji into radicals during inference stage in fallback mode.

_Fallback mode_ - when the kanji is not found in WaniKani subjects, the app uses KanjiAlive and KRAD data to prepare prompts (if possible).

---

- Model [card](https://huggingface.co/elna4os/mnemonify) on Hugging Face
- Hugging Face Spaces [demo](https://huggingface.co/spaces/elna4os/mnemonify)

---

<ins>To do</ins>:

- Find a reliable and open-source way to decompose kanji into radicals (unfortunately, WaniKani license doesn't allow to publicly share their radicals data). KRAD decomposition is too atomic which apparently is not really good for mnemonics.
