### Generate kanji mnemonics with fine-tuned Phi-3-Mini-4K-Instruct

---

<ins>Status</ins>: development

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

---

<ins>Data sources and their usage</ins>:

- Subjects from [WaniKani API](https://docs.api.wanikani.com/20170710/#introduction): kanji (2083), vocabulary (6650). Used only to fine-tune the model (as i understand, WaniKani's terms of service don't allow to publicly share data fetched from API).
- [KanjiAlive](https://github.com/kanjialive/kanji-data-media) offline data: radicals and kanji. Used to prepare prompts during inference stage.

---

<ins>To do</ins>:

- Retrain model with the new prompts
- Merge/unload -> convert to GGUF -> quantize to 4-bit
- User request validation
- Inference prompt generator (from KanjiAlive data)
- Host model on Hugging Face Spaces (Streamlit + llama.cpp) and set up simple logging (Python Telegram Bot for ex.)
