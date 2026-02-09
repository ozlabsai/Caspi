this file is only a suggestion for a course of action. it is NOT binding. you can suggest/use other ideas or methods.

## What data are we using?

### Use

1. **`ivrit-ai/crowd-transcribe-v5` as the main train set**
   It’s the big Hebrew crowd ASR corpus, but note: it’s **gated on HF** (you must accept the ivrit.ai license / share contact info to access the files). ([Hugging Face][1])
   ➡️ Once you accept it, we’ll inspect the exact columns and build filters.

2. **`ivrit-ai/crowd-recital-whisper-training` as a robustness add-on** (small, but valuable)
   It’s explicitly Hebrew and includes the schema: 16kHz audio + `transcript`, and sometimes the transcript includes **Whisper-style timestamp tokens** when `has_timestamps=true`. ([Hugging Face][2])
   ➡️ For Qwen3-ASR training, you’ll almost certainly want to **strip those timestamp tokens** unless you intentionally want the model to emit them.

3. maybe we should also add syntethic augmentations / adv audio - like compressed, overlapped (take to audio sample, one should be really quiet etc...)

### Don’t use (for Hebrew)

* **`crowd-whatsapp-yi`** (and the other `*-yi*` datasets): these are **Yiddish**, not Hebrew. ([Hugging Face][3])

---

## Fine-tuning plan (full model output)

### Phase 0 — Decide your target behavior (1 hour decision that saves days)

* **Output format**: do you want “clean Hebrew text” (recommended) or punctuation-heavy?
* **Normalization rules**: numbers (digits vs spelled), apostrophes/geresh, maqaf/hyphen style, optional diacritics (usually **no niqqud**).

This matters because ASR models will “learn your orthography”.

---

## Phase 1 — Build a “gold” training table (data engineering)

### 1) Load + standardize

* Resample all audio to **16kHz** (recital already is 16kHz). ([Hugging Face][2])
* Create a unified table:

  * `audio` (wav/array)
  * `text` (normalized transcript)
  * `duration_sec`
  * `source` (v5 vs recital)
  * `quality_flags` (whatever v5 provides once you can read it)

### 2) Clean transcripts

* Remove:

  * timestamp tokens like `<|0.00|>`… when present ([Hugging Face][2])
  * obvious junk: duplicated punctuation, “um/eh” artifacts if inconsistent
* Keep consistent:

  * quotes, parentheses, dash/maqaf style
  * English words inside Hebrew (decide policy)

### 3) Duration filtering + bucketing (huge for compute)

* Set a hard max (e.g. **≤20–30s** per clip for training) at first.
* Bucket by duration so you don’t waste padding.

This is one of the biggest levers for making “1.7B” feel cheap.

---

## Phase 2 — Format examples in the model’s expected ASR style

Qwen3-ASR is shipped via the `qwen-asr` toolkit and supports language ID + ASR; the model card also shows a structured API where audio is the user content and the model returns parseable ASR output (`parse_asr_output`). ([Hugging Face][4])

**Key detail:** the model card’s explicit language list doesn’t mention Hebrew, even though they claim 52 languages/dialects overall. ([Hugging Face][4])
So: during finetuning you’ll likely want the training target to be **just the Hebrew transcript** (and let language detection be implicit), unless the official training format expects a language tag (we can verify in the repo/examples once you want). ([GitHub][5])

---

## Phase 3 — Training setup (full checkpoint)

### Baseline (stable, proven)

* **bf16**, FlashAttention if available ([Hugging Face][4])
* **Gradient checkpointing**
* **Deepspeed ZeRO-3 or FSDP** if you want to update many layers without OOM (especially with Adam-type states). ZeRO-3 is a standard approach for sharding optimizer+grads. ([arXiv][6])

### Freezing strategy (still outputs a full model)

Even if you insist on “full model output”, you don’t have to update every weight:

* **Stage A (fast + safe):** train projector/adaptor + top decoder block(s)
* **Stage B (only if needed):** unfreeze more decoder layers
* **Stage C (rare):** touch audio encoder if you see consistent acoustic confusions

You still save a single merged checkpoint at the end; it’s just that some layers stayed unchanged.

---

## Phase 4 — Evaluation loop (don’t skip)

* Use **held-out** from v5 + the recital test split (or carve one).
* Report **WER + CER**.
* Also run a “stress set”:

  * noisy clips
  * fast speech
  * proper nouns
  * mixed Hebrew/English

(If you care about names/jargon later, Qwen3-ASR also supports “context prompting” at inference time for biasing outputs. ([vLLM][7]))

---

# Optimizer choice: AdamW vs Muon (bleeding edge)

### Why AdamW is still the default for fine-tuning

* It’s the most battle-tested in HF/DeepSpeed/FSDP pipelines.
* It’s predictable for **post-training / SFT**, where you care about stability and avoiding catastrophic drift.

### Can we use Muon?

Yes — but treat it as an **experiment track**, not your only run.

Muon has credible evidence of speed/efficiency gains in some LLM training settings (e.g., Keller Jordan’s writeup with comparisons). ([kellerjordan.github.io][8])
There are also newer variants like **NorMuon** claiming better efficiency and practical distributed implementations. ([OpenReview][9])

**Practical caveats (for your case):**

* Tooling maturity: Muon isn’t as “plug-and-play” as AdamW inside standard HF Trainer + DeepSpeed stacks.
* Evidence base: a lot of Muon wins are reported for **training regimes**; for **ASR SFT adaptation**, results may vary.
* Model structure: Qwen3-ASR isn’t a plain text-only transformer; you’re training a multimodal stack.

### Recommended optimizer plan

* **Run 1 (baseline): AdamW** — establish a strong reference WER/CER.
* **Run 2 (experimental): Muon / NorMuon** — same data, same schedule, measure:

  * wallclock to target WER
  * stability (spikes, regressions)
  * final WER on stress set

If Muon wins, awesome — you keep it. If it’s flaky, you didn’t burn the whole project.
