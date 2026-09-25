"""
Self-contained inference module for the Optical+SAR VQA pipeline.
Requires: torch, transformers, rasterio, einops (pip install rasterio einops)

Usage:
    from inference import load_pipeline, predict
    pipeline = load_pipeline(
        croma_py_path="pretrain_croma.py",
        croma_ckpt_path="CROMA_base.pt",
        croma_config_path="croma_config.json",
        projector_ckpt_path="projector_final.pt",
        qwen_dir="qwen_final",
    )
    answer = predict(pipeline, s1_dir, s2_dir, sar_id, optical_id, "Is water present?")
"""

import os
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import rasterio
from rasterio.enums import Resampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- CROMA loading ----------

def _load_croma_class(croma_py_path):
    spec = importlib.util.spec_from_file_location("pretrain_croma", croma_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CROMA


def load_croma_model(croma_py_path, croma_ckpt_path, croma_config):
    CROMA = _load_croma_class(croma_py_path)
    model = CROMA(**croma_config)
    checkpoint = torch.load(croma_ckpt_path, map_location=device)

    # Checkpoint is five separate sub-module state-dicts, not one flat state_dict.
    KEY_PREFIX_MAP = {
        "s1_encoder": "radar_encoder",
        "s2_encoder": "optical_encoder",
        "joint_encoder": "cross_encoder",
        "s1_GAP_FFN": "GAP_FFN_radar",
        "s2_GAP_FFN": "GAP_FFN_optical",
    }
    flat_state_dict = {}
    for ckpt_key, model_prefix in KEY_PREFIX_MAP.items():
        if ckpt_key not in checkpoint:
            continue
        for sub_key, tensor in checkpoint[ckpt_key].items():
            flat_state_dict[f"{model_prefix}.{sub_key}"] = tensor

    model.load_state_dict(flat_state_dict, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def croma_joint_embedding(croma_model, s1_batch, s2_batch):
    """s1_batch: (B, 2, 120, 120) VV/VH. s2_batch: (B, 12, 120, 120). Returns (B, 225, 768)."""
    s1_batch = s1_batch.to(device)
    s2_batch = s2_batch.to(device)
    attn_bias = croma_model.attn_bias.to(device)
    return croma_model.forward_inference(s1_batch, s2_batch, attn_bias) \
        if hasattr(croma_model, "forward_inference") \
        else _manual_joint_forward(croma_model, s1_batch, s2_batch, attn_bias)


def _manual_joint_forward(croma_model, s1_batch, s2_batch, attn_bias):
    # Mirrors CROMA.forward()'s encoder half, skipping MAE masking + losses.
    radar_tokens = croma_model.radar_encoder(s1_batch, attn_bias=attn_bias, mask_info=None)
    optical_tokens = croma_model.optical_encoder(s2_batch, attn_bias=attn_bias, mask_info=None)
    joint = croma_model.cross_encoder(radar_tokens, optical_tokens)
    return joint


# ---------- Projector ----------

class Projector(nn.Module):
    def __init__(self, croma_dim, qwen_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or (croma_dim + qwen_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(croma_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, qwen_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Preprocessing ----------

def s1_acquisition_id_from_patch_id(patch_id):
    parts = patch_id.split("_")
    return "_".join(parts[:-3])


def s2_acquisition_id_from_patch_id(patch_id):
    parts = patch_id.split("_")
    return "_".join(parts[:-2])


def band_file_path(root_dir, patch_id, band, sensor):
    if sensor == "s1":
        acq_id = s1_acquisition_id_from_patch_id(patch_id)
    else:
        acq_id = s2_acquisition_id_from_patch_id(patch_id)
    return os.path.join(root_dir, acq_id, patch_id, f"{patch_id}_{band}.tif")


def read_single_band(path, target_size=None, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        if target_size is not None:
            arr = src.read(1, out_shape=(target_size, target_size), resampling=resampling)
        else:
            arr = src.read(1)
    return arr.astype(np.float32)


def load_multiband(root_dir, patch_id, bands, sensor, target_size=None):
    arrays = [read_single_band(band_file_path(root_dir, patch_id, b, sensor), target_size) for b in bands]
    return np.stack(arrays, axis=0)


S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def load_patch_pair(s1_dir, s2_dir, sar_id, optical_id, patch_size=120):
    s1 = load_multiband(s1_dir, sar_id, S1_BANDS, "s1", target_size=patch_size)
    s2 = load_multiband(s2_dir, optical_id, S2_BANDS, "s2", target_size=patch_size)
    return torch.from_numpy(s1).unsqueeze(0), torch.from_numpy(s2).unsqueeze(0)


# ---------- Full pipeline ----------

class VQAPipeline:
    def __init__(self, croma_model, projector, qwen_model, tokenizer):
        self.croma_model = croma_model
        self.projector = projector
        self.qwen_model = qwen_model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_answer(self, s1, s2, question, max_new_tokens=16):
        joint = croma_joint_embedding(self.croma_model, s1, s2)
        visual_tokens = self.projector(joint)

        q_enc = self.tokenizer(question, return_tensors="pt").to(device)
        q_embeds = self.qwen_model.get_input_embeddings()(q_enc["input_ids"])

        inputs_embeds = torch.cat([visual_tokens, q_embeds], dim=1)
        attn_mask = torch.cat(
            [torch.ones(visual_tokens.shape[:2], device=device, dtype=q_enc["attention_mask"].dtype),
             q_enc["attention_mask"]], dim=1,
        )
        gen_ids = self.qwen_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def load_pipeline(croma_py_path, croma_ckpt_path, croma_config_path,
                   projector_ckpt_path, qwen_dir):
    with open(croma_config_path) as f:
        croma_config = json.load(f)

    croma_model = load_croma_model(croma_py_path, croma_ckpt_path, croma_config)

    tokenizer = AutoTokenizer.from_pretrained(qwen_dir)
    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_dir, torch_dtype=torch.float32).to(device)
    qwen_model.eval()

    projector = Projector(croma_config["encoder_dim"], qwen_model.config.hidden_size).to(device)
    projector.load_state_dict(torch.load(projector_ckpt_path, map_location=device))
    projector.eval()

    return VQAPipeline(croma_model, projector, qwen_model, tokenizer)


def predict(pipeline, s1_dir, s2_dir, sar_id, optical_id, question, patch_size=120):
    s1, s2 = load_patch_pair(s1_dir, s2_dir, sar_id, optical_id, patch_size)
    return pipeline.generate_answer(s1, s2, question)
