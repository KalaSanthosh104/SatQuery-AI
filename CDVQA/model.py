import torch
import torch.nn as nn


class CDVQAModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_question_types=8,
        num_categories=6,
        num_ratios=11
    ):

        super().__init__()

        # ----------------------------------------------------
        # Visual encoder
        # ----------------------------------------------------

        self.visual_encoder = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ----------------------------------------------------
        # Question encoder
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.question_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # ----------------------------------------------------
        # Question type embedding
        # ----------------------------------------------------

        self.question_type_embedding = nn.Embedding(
            num_question_types,
            32
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(416, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Output heads
        # ----------------------------------------------------

        self.binary_head = nn.Linear(
            128,
            2
        )

        self.category_head = nn.Linear(
            128,
            num_categories
        )

        self.ratio_head = nn.Linear(
            128,
            num_ratios
        )

    def encode_image(self, image):

        features = self.visual_encoder(image)

        features = features.view(
            features.size(0),
            -1
        )

        return features

    def encode_question(
        self,
        input_ids,
        attention_mask
    ):

        embedded = self.embedding(input_ids)

        output, (hidden, cell) = self.question_lstm(
            embedded
        )

        question_features = hidden[-1]

        return question_features

    def forward(
        self,
        image1,
        image2,
        input_ids,
        attention_mask,
        question_type_id
    ):

        # Image features
        image1_features = self.encode_image(image1)
        image2_features = self.encode_image(image2)

        # Absolute difference between T1 and T2
        image_difference = torch.abs(
            image2_features - image1_features
        )

        # Question features
        question_features = self.encode_question(
            input_ids,
            attention_mask
        )

        # Question type features
        question_type_features = self.question_type_embedding(
            question_type_id
        )

        # Fuse everything
        fused_features = torch.cat(
            [
                image_difference,
                question_features,
                question_type_features
            ],
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )

        # Output
        binary_output = self.binary_head(
            fused_features
        )

        category_output = self.category_head(
            fused_features
        )

        ratio_output = self.ratio_head(
            fused_features
        )

        return {
            "binary": binary_output,
            "category": category_output,
            "ratio": ratio_output
        }