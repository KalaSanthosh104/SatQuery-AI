import math
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer

from model import RSVQAModel


class RSVQAInference:
    """
    Inference wrapper for the trained SatQuery RSVQA-LR model.

    Input:
        image_path: path to a satellite image
        question: natural-language question

    Output:
        dictionary containing question type, answer and confidence
        (confidence is not provided for count regression).
    """

    def __init__(
        self,
        checkpoint_path="best_count_finetuned_rsvqa.pt",
        vit_name="google/vit-base-patch16-224-in21k",
        bert_name="bert-base-uncased",
        max_length=32,
        device=None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length

        print(f"Using device: {self.device}")

        # Build the exact architecture used during training.
        self.model = RSVQAModel(vit_name=vit_name, bert_name=bert_name)

        # Load checkpoint and handle the Transformers state-key naming
        # differences used by the saved checkpoint.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")

        state_dict = self._convert_checkpoint_keys(state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch.\n"
                f"Missing keys: {missing}\n"
                f"Unexpected keys: {unexpected}"
            )

        self.model.to(self.device)
        self.model.eval()

        self.image_processor = ViTImageProcessor.from_pretrained(vit_name)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print("RSVQA model loaded successfully!")

    @staticmethod
    def _convert_checkpoint_keys(state_dict):
        """
        Convert saved ViT/BERT parameter names to the names expected by
        the current model implementation.
        """
        converted = {}

        for key, value in state_dict.items():
            new_key = key

            # Apply mappings separately because ViT and BERT use
            # different parameter naming conventions.
            if new_key.startswith("vit."):
                new_key = new_key.replace(
                    "vit.encoder.layer.", "vit.layers."
                )
                new_key = new_key.replace(
                    ".attention.attention.query.", ".attention.q_proj."
                )
                new_key = new_key.replace(
                    ".attention.attention.key.", ".attention.k_proj."
                )
                new_key = new_key.replace(
                    ".attention.attention.value.", ".attention.v_proj."
                )
                new_key = new_key.replace(
                    ".attention.output.dense.", ".attention.o_proj."
                )
                new_key = new_key.replace(
                    ".intermediate.dense.", ".mlp.fc1."
                )
                new_key = new_key.replace(
                    ".output.dense.", ".mlp.fc2."
                )

            elif new_key.startswith("bert."):
                new_key = new_key.replace(
                    ".attention.o_proj.", ".attention.output.dense."
                )
                new_key = new_key.replace(
                    ".mlp.fc1.", ".intermediate.dense."
                )
                new_key = new_key.replace(
                    ".mlp.fc2.", ".output.dense."
                )

            converted[new_key] = value

        return converted

    @staticmethod
    def detect_question_type(question):
        q = question.lower().strip()

        if (
            "how many" in q
            or "number of" in q
            or "count" in q
        ):
            return "count"

        if "rural" in q or "urban" in q:
            return "rural_urban"

        if any(
            word in q
            for word in [
                "more",
                "less",
                "equal",
                "same",
                "greater",
                "fewer",
            ]
        ):
            return "comp"

        return "presence"

    def predict(self, image_path, question):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt",
        )
        pixel_values = image_inputs["pixel_values"].to(self.device)

        text_inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        question_type = self.detect_question_type(question)

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if question_type == "presence":
            probs = torch.softmax(outputs["presence_logits"], dim=1)[0]
            label = int(torch.argmax(probs).item())
            answer = "yes" if label == 1 else "no"
            confidence = float(probs[label].item()) * 100

        elif question_type == "comp":
            probs = torch.softmax(outputs["comp_logits"], dim=1)[0]
            label = int(torch.argmax(probs).item())
            answer = "yes" if label == 1 else "no"
            confidence = float(probs[label].item()) * 100

        elif question_type == "rural_urban":
            probs = torch.softmax(outputs["rural_urban_logits"], dim=1)[0]
            label = int(torch.argmax(probs).item())
            answer = "urban" if label == 1 else "rural"
            confidence = float(probs[label].item()) * 100

        elif question_type == "count":
            count_log = float(outputs["count_log"][0].item())
            answer = max(0, round(math.expm1(count_log)))
            confidence = None

        else:
            raise ValueError(f"Unsupported question type: {question_type}")

        return {
            "question_type": question_type,
            "answer": answer,
            "confidence": confidence,
        }


if __name__ == "__main__":
    print("\n===== RSVQA Interactive Inference =====")

    checkpoint = input(
        "Checkpoint path [best_count_finetuned_rsvqa.pt]: "
    ).strip()

    if not checkpoint:
        checkpoint = "best_count_finetuned_rsvqa.pt"

    image_path = input("Enter image path: ").strip()
    question = input("Enter question: ").strip()

    predictor = RSVQAInference(checkpoint_path=checkpoint)
    result = predictor.predict(image_path, question)

    print("\n===== RESULT =====")
    print("Question type :", result["question_type"])
    print("Answer        :", result["answer"])

    if result["confidence"] is not None:
        print(f"Confidence    : {result['confidence']:.2f}%")
    else:
        print("Confidence    : N/A (count regression)")
