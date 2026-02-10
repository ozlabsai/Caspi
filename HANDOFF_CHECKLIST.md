# Round 2 Training - Handoff Checklist

**Purpose:** Verify all components are ready before GPU team begins execution
**Target:** GPU Training Team Lead
**Date:** 2026-02-10

---

## Pre-Handoff Verification

### Code and Repository

- [ ] **Repository Status**
  - [ ] All Round 2 code committed to repository
  - [ ] No uncommitted changes in critical files
  - [ ] Git tag created: `round2-v1.0`

- [ ] **Critical Files Present**
  - [ ] `train_round2_gradual.py` (main entry point)
  - [ ] `train_hebrew_asr_enhanced.py` (core training logic)
  - [ ] `scripts/phase0_align_audit.py` (data audit)
  - [ ] `scripts/eval_round2.py` (evaluation)
  - [ ] `scripts/setup_wandb.sh` (W&B configuration)
  - [ ] `pyproject.toml` (dependencies with wandb)

- [ ] **Documentation Complete**
  - [ ] `DEPLOYMENT_ROUND2.md` (35-page detailed guide)
  - [ ] `QUICK_START_ROUND2.md` (2-page quick reference)
  - [ ] `TECHNICAL_SPEC_ROUND2.md` (25-page technical details)
  - [ ] `wandb_setup.md` (experiment tracking guide)
  - [ ] `CLAUDE.md` updated with Round 2 commands

- [ ] **Code Quality**
  - [ ] Python syntax validated (`python -m py_compile`)
  - [ ] No import errors when loading modules
  - [ ] All required dependencies in `pyproject.toml`

### Environment and Access

- [ ] **GPU Access**
  - [ ] Lambda Labs / RunPod / HF Spaces account created
  - [ ] 2x A100 (40GB) instance type available
  - [ ] Billing configured and approved ($50 budget)
  - [ ] SSH keys / access credentials provided to team

- [ ] **HuggingFace Access**
  - [ ] Team account/token with read access to:
    - [ ] `Qwen/Qwen3-ASR-1.7B`
    - [ ] `Qwen/Qwen3-ForcedAligner-0.6B`
    - [ ] `ivrit-ai/crowd-transcribe-v5`
    - [ ] `ivrit-ai/crowd-recital-whisper-training`
  - [ ] Write access to target namespace for model upload
  - [ ] Token added to team secrets/vault

- [ ] **Weights & Biases Access**
  - [ ] Team account created (free tier)
  - [ ] API keys distributed to team
  - [ ] Project `qwen3-asr-hebrew` created
  - [ ] Team members added (if using team workspace)

### Pre-Execution Testing

- [ ] **Dependency Installation (Local)**
  ```bash
  cd caspi && uv sync
  # Verify no errors
  ```

- [ ] **Import Testing (Local)**
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}')"
  python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
  python -c "import wandb; print(f'W&B: {wandb.__version__}')"
  python -c "from qwen_asr import Qwen3ForcedAligner; print('Qwen-ASR OK')"
  ```

- [ ] **Phase 0 Dry Run (Optional, Local/CPU)**
  - [ ] Run on small sample (100 samples) to verify pipeline
  - [ ] Check output format of `alignment_report.json`
  - [ ] Verify decision gate logic works

---

## GPU Team Onboarding

### Knowledge Transfer

- [ ] **Training Session Scheduled**
  - [ ] Date/time: _______________
  - [ ] Attendees: _______________
  - [ ] Location: _______________

- [ ] **Documentation Review**
  - [ ] Team read `DEPLOYMENT_ROUND2.md` (or `QUICK_START_ROUND2.md`)
  - [ ] Technical lead reviewed `TECHNICAL_SPEC_ROUND2.md`
  - [ ] Questions documented and answered

- [ ] **Runbook Walkthrough**
  - [ ] Phase 0 execution demonstrated
  - [ ] Decision gate process explained
  - [ ] Training launch demonstrated
  - [ ] Monitoring dashboard shown (W&B)
  - [ ] Troubleshooting scenarios discussed

- [ ] **Emergency Procedures**
  - [ ] OOM handling procedure reviewed
  - [ ] Graceful stop procedure practiced
  - [ ] Rollback plan understood
  - [ ] Escalation contacts provided

### Success Criteria Review

- [ ] **Primary Metrics Understood**
  - [ ] WER improvement: 12.3% → 10.5-11.0%
  - [ ] Minimum acceptable: <11.5%
  - [ ] Target per-domain improvements reviewed

- [ ] **Phase 0 Decision Gate**
  - [ ] <10% low quality → PROCEED
  - [ ] 10-15% low quality → CAUTION
  - [ ] >15% low quality → STOP

- [ ] **Training Stability Indicators**
  - [ ] GPU memory: 28-34 GB (safe range)
  - [ ] Eval WER: should decrease
  - [ ] Strategy switch at epoch 3: expected small spike

---

## Pre-Launch Checklist (GPU Team)

### Hardware Verification

- [ ] **GPU Instance Provisioned**
  - [ ] Provider: _______________
  - [ ] Instance type: 2x A100 (40GB)
  - [ ] Hourly cost: $______/hour
  - [ ] Region: _______________

- [ ] **GPU Health Check**
  ```bash
  nvidia-smi
  # Verify:
  # - 2 GPUs visible
  # - Each 40 GB memory
  # - Driver version OK
  # - CUDA version >= 11.8
  ```

- [ ] **Disk Space**
  ```bash
  df -h
  # Verify at least 150 GB free
  ```

- [ ] **Network Connectivity**
  ```bash
  curl -I https://huggingface.co
  curl -I https://api.wandb.ai
  # Both should return HTTP 200
  ```

### Software Setup

- [ ] **Repository Cloned**
  ```bash
  git clone <repository-url>
  cd caspi
  git checkout round2-v1.0  # Or specific commit
  ```

- [ ] **Dependencies Installed**
  ```bash
  uv sync
  # Verify completion: echo $?  # Should be 0
  ```

- [ ] **Logins Completed**
  ```bash
  huggingface-cli login
  wandb login
  ```

- [ ] **Environment Variables Set**
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0,1
  export WANDB_PROJECT="qwen3-asr-hebrew"
  export WANDB_RUN_NAME="round2-production-$(date +%Y%m%d)"
  export WANDB_PHASE0_LOGGING="true"
  ```

### Pre-Download (Recommended)

- [ ] **Model Downloaded**
  ```bash
  python -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('Qwen/Qwen3-ASR-1.7B', trust_remote_code=True); print('Model OK')"
  ```

- [ ] **Datasets Downloaded**
  ```bash
  python -c "from datasets import load_dataset; load_dataset('ivrit-ai/crowd-transcribe-v5', split='train'); print('Dataset 1 OK')"
  python -c "from datasets import load_dataset; load_dataset('ivrit-ai/crowd-recital-whisper-training', split='train'); print('Dataset 2 OK')"
  ```

- [ ] **Forced Aligner Downloaded**
  ```bash
  python -c "from qwen_asr import Qwen3ForcedAligner; Qwen3ForcedAligner.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B'); print('Aligner OK')"
  ```

---

## Phase 0 Launch Checklist

- [ ] **Command Ready**
  ```bash
  uv run python scripts/phase0_align_audit.py
  ```

- [ ] **Monitoring Prepared**
  - [ ] Terminal logs captured: `tee phase0.log`
  - [ ] Start time recorded: _______________
  - [ ] Expected completion: _______________ (~2-3 hours)

- [ ] **Execution**
  - [ ] Command executed
  - [ ] No immediate errors
  - [ ] Progress bar showing

- [ ] **Completion Verification**
  - [ ] `phase0_audit_results/alignment_report.json` exists
  - [ ] Decision field populated: _______________
  - [ ] Low quality percentage: _______________

---

## Decision Gate (After Phase 0)

- [ ] **Results Reviewed**
  ```bash
  cat phase0_audit_results/alignment_report.json | jq '.'
  ```

- [ ] **Decision Made**
  - [ ] ✅ PROCEED: Low quality <10%, continue to training
  - [ ] ⚠️ CAUTION: Low quality 10-15%, proceed with monitoring
  - [ ] ❌ STOP: Low quality >15%, escalate to data team

- [ ] **Sign-Off Recorded**
  ```bash
  echo "Phase 0 Decision: _______" >> phase0_signoff.txt
  echo "Reviewed by: _______" >> phase0_signoff.txt
  echo "Date: $(date)" >> phase0_signoff.txt
  ```

---

## Training Launch Checklist (Only if Phase 0 PROCEED)

- [ ] **Pre-Flight Checks**
  - [ ] Phase 0 completed successfully
  - [ ] Decision gate passed (PROCEED or CAUTION)
  - [ ] GPU memory baseline: `nvidia-smi` (should be ~5-10 GB used)
  - [ ] Disk space sufficient: >150 GB free

- [ ] **Monitoring Setup**
  - [ ] W&B dashboard open: https://wandb.ai/{username}/qwen3-asr-hebrew
  - [ ] TensorBoard started (backup): `tensorboard --logdir qwen3-asr-hebrew-round2`
  - [ ] `nvidia-smi` watch running: `watch -n 5 nvidia-smi`

- [ ] **Training Command**
  ```bash
  uv run python train_round2_gradual.py 2>&1 | tee training.log
  ```

- [ ] **Initial Verification (First 15 minutes)**
  - [ ] Model loading successful
  - [ ] Strategy B freezing applied
  - [ ] LoRA setup completed
  - [ ] Datasets loaded
  - [ ] W&B run initialized
  - [ ] First training step completed
  - [ ] GPU memory: 28-32 GB (Strategy B)
  - [ ] No OOM errors

- [ ] **Checkpoint 1 (After 2 hours / Epoch 1)**
  - [ ] Eval WER decreasing (should be ~16-18%)
  - [ ] GPU memory stable
  - [ ] checkpoint-500 saved
  - [ ] No warnings in logs

- [ ] **Checkpoint 2 (After 5 hours / Epoch 2-3 boundary)**
  - [ ] Console shows "Switching to Strategy A"
  - [ ] GPU memory increases to 32-34 GB (expected)
  - [ ] Eval WER ~13%
  - [ ] checkpoint-1000 saved

- [ ] **Checkpoint 3 (After 8 hours / Epoch 4)**
  - [ ] Eval WER continuing to decrease (~11-12%)
  - [ ] GPU memory stable at 33-34 GB
  - [ ] No OOM or divergence

- [ ] **Final Verification (After 12 hours / Epoch 5)**
  - [ ] Training completed successfully
  - [ ] Final WER: _______________ (target: 10.5-11.0%)
  - [ ] Final CER: _______________ (target: 4.5-5.0%)
  - [ ] Model saved to `./qwen3-asr-hebrew-round2/`
  - [ ] Pushed to HuggingFace Hub: _______________
  - [ ] All checkpoints preserved

---

## Post-Training Checklist

- [ ] **Results Validation**
  - [ ] Final WER meets minimum criteria (<11.5%)
  - [ ] Model file sizes reasonable (~17 GB)
  - [ ] `config.json` and `tokenizer_config.json` present
  - [ ] Model loads without errors:
    ```bash
    python -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('./qwen3-asr-hebrew-round2', trust_remote_code=True); print('Model OK')"
    ```

- [ ] **Artifacts Collected**
  - [ ] Training logs: `training.log`
  - [ ] Phase 0 report: `phase0_audit_results/alignment_report.json`
  - [ ] W&B run URL: _______________
  - [ ] Model Hub URL: _______________
  - [ ] All checkpoints backed up

- [ ] **Evaluation (Optional)**
  ```bash
  uv run python scripts/eval_round2.py \
      --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \
      --round2-model ./qwen3-asr-hebrew-round2
  ```

- [ ] **Cost Verification**
  - [ ] Total GPU hours: _______________
  - [ ] Total cost: $_______________ (expected: ~$48)
  - [ ] Within budget: [ ] Yes [ ] No

---

## Handoff Completion

### Sign-Off

**ML Engineer (Code Author):**
- Name: _______________
- Date: _______________
- Signature: _______________
- Notes: _______________

**GPU Team Lead (Recipient):**
- Name: _______________
- Date: _______________
- Signature: _______________
- Confirmed understanding: [ ] Yes [ ] No
- Questions resolved: [ ] Yes [ ] No

### Post-Handoff

- [ ] **Follow-Up Meeting Scheduled**
  - [ ] Date/time: _______________
  - [ ] Agenda: Review results, discuss next steps

- [ ] **Results Report Due Date**
  - [ ] Target: _______________ (suggest: 3 days after training completion)
  - [ ] Format: Summary email + W&B link

- [ ] **Issue Tracking**
  - [ ] GitHub issues created for any blockers
  - [ ] Slack channel: _______________
  - [ ] Primary contact: _______________

---

## Emergency Contacts

**During Business Hours:**
- Project Lead: [Name] - [Email/Slack/Phone]
- ML Engineer: [Name] - [Email/Slack/Phone]
- Data Team: [Name] - [Email/Slack/Phone]

**After Hours / Emergencies:**
- On-Call: [Phone/PagerDuty]
- Escalation: [Manager contact]

**Vendor Support:**
- Lambda Labs: support@lambdalabs.com
- HuggingFace: support@huggingface.co
- W&B: support@wandb.ai

---

## Appendix: Quick Reference

### Most Important Commands

```bash
# Phase 0
uv run python scripts/phase0_align_audit.py

# Check Phase 0 results
cat phase0_audit_results/alignment_report.json | jq '.decision'

# Training
uv run python train_round2_gradual.py

# Monitor GPU
watch -n 5 nvidia-smi

# Check W&B
open "https://wandb.ai/$(wandb whoami)/qwen3-asr-hebrew"

# Emergency stop
# Press Ctrl+C ONCE, wait for checkpoint save
```

### Most Important Files

- `DEPLOYMENT_ROUND2.md` - Full deployment guide
- `QUICK_START_ROUND2.md` - Quick reference
- `TECHNICAL_SPEC_ROUND2.md` - Technical details
- `phase0_audit_results/alignment_report.json` - Data quality results
- `training.log` - Training execution log

### Success Criteria Reminder

- ✅ Phase 0: <10% low quality
- ✅ Training: Completes without OOM
- ✅ WER: <11.5% minimum, 10.5-11.0% target
- ✅ Cost: ~$48 (within $50 budget)

---

**Handoff Status:** [ ] Complete [ ] Incomplete

**Ready to Execute:** [ ] Yes [ ] No

**If No, blockers:** _______________
