
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel


class RSVQAModel(nn.Module):

    def __init__(
        self,
        vit_name="google/vit-base-patch16-224-in21k",
        bert_name="bert-base-uncased"
    ):
        super().__init__()

        # -------------------------
        # Image encoder: ViT
        # -------------------------
        self.vit = ViTModel.from_pretrained(vit_name)

        self.image_dim = self.vit.config.hidden_size

        # -------------------------
        # Text encoder: BERT
        # -------------------------
        self.bert = AutoModel.from_pretrained(bert_name)

        self.text_dim = self.bert.config.hidden_size

        # -------------------------
        # Multimodal fusion
        # -------------------------
        self.fusion = nn.Sequential(
            nn.Linear(
                self.image_dim + self.text_dim,
                768
            ),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                768,
                512
            ),
            nn.ReLU(),

            nn.Dropout(0.2)
        )

        # -------------------------
        # Task-specific heads
        # -------------------------

        # Presence: yes / no
        self.presence_head = nn.Linear(
            512,
            2
        )

        # Comparison: yes / no
        self.comp_head = nn.Linear(
            512,
            2
        )

        # Rural / Urban
        self.rural_urban_head = nn.Linear(
            512,
            2
        )

        # Count: numerical prediction
        self.count_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1)
        )

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask
    ):

        # -------------------------
        # ViT image features
        # -------------------------
        vit_output = self.vit(
            pixel_values=pixel_values
        )

        image_features = vit_output.last_hidden_state[:, 0]

        # -------------------------
        # BERT text features
        # -------------------------
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = bert_output.last_hidden_state[:, 0]

        # -------------------------
        # Combine image + question
        # -------------------------
        combined_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        # -------------------------
        # Fusion
        # -------------------------
        fused_features = self.fusion(
            combined_features
        )

        # -------------------------
        # Predictions
        # -------------------------
        presence_logits = self.presence_head(
            fused_features
        )

        comp_logits = self.comp_head(
            fused_features
        )

        rural_urban_logits = self.rural_urban_head(
            fused_features
        )

        # Predict log1p(count)
        count_log = self.count_head(
            fused_features
        ).squeeze(1)

        return {
            "presence_logits": presence_logits,
            "comp_logits": comp_logits,
            "rural_urban_logits": rural_urban_logits,
            "count_log": count_log
        }
