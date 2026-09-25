# Optical + SAR VQA Pipeline — Handoff Package

Pipeline: Sentinel-1 + Sentinel-2 -> CROMA (frozen) -> Projector -> Qwen -> answer.

## Files
- `CROMA_base.pt` — frozen pretrained CROMA backbone checkpoint (not trained by us, upstream weights).
- `pretrain_croma.py` — CROMA model class definition (upstream source).
- `croma_config.json` — exact config dict CROMA_base.pt was built/loaded with. Must match exactly or the checkpoint will not load correctly.
- `projector_final.pt` — our trained projector weights (maps CROMA's 768-dim joint embedding into Qwen's hidden size).
- `qwen_final/` — fine-tuned Qwen model + tokenizer, ready for `AutoModelForCausalLM.from_pretrained()`.
- `inference.py` — self-contained module with `load_pipeline()` and `predict()`. No training code, no Kaggle-specific paths. Drop this straight into the backend.

## How to use it (backend side)
```python
from inference import load_pipeline, predict

pipeline = load_pipeline(
    croma_py_path="CROMA_base.pt".replace("CROMA_base.pt", "pretrain_croma.py"),
    croma_ckpt_path="CROMA_base.pt",
    croma_config_path="croma_config.json",
    projector_ckpt_path="projector_final.pt",
    qwen_dir="qwen_final",
)

answer = predict(
    pipeline,
    s1_dir="/path/to/BigEarthNet-S1",
    s2_dir="/path/to/BigEarthNet-S2",
    sar_id="S1B_IW_GRDH_1SDV_20180130T162407_34TEP_6_85",
    optical_id="S2A_..._34TEP_6_85",  # must be the MATCHING optical patch id
    question="Is water present in this patch?",
)
print(answer)
```

## Requirements
```
pip install torch transformers rasterio einops
```

## Notes for whoever loads this
- CROMA and Qwen are both **frozen** in this package — only the projector was trained (Stage 1).
  If Stage 2 (LoRA on Qwen/CROMA) was also run, ask whoever sent this whether `qwen_final/`
  includes the merged LoRA weights or needs a separate adapter loaded.
- `sar_id` and `optical_id` must refer to the SAME geographic patch, just from different
  sensors — they will NOT have identical filenames, only a shared spatial/temporal reference.
- Patch size is fixed at 120x120 pixels (matches CROMA's `num_patches=225` config, i.e. a 15x15
  grid of 8px patches). Feeding a different resolution will silently mismatch the joint encoder.
