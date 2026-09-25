
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor, AutoTokenizer
from dataset import RSVQADataset


class RSVQAModelDataset(Dataset):

    def __init__(self, dataset_root, split="train", max_length=32):

        self.dataset = RSVQADataset(
            dataset_root=dataset_root,
            split=split
        )

        self.image_dir = (
            dataset_root + "/Images_LR/Images_LR"
        )

        self.image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        self.max_length = max_length


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        sample = self.dataset[index]

        # --------------------------------------------------
        # IMPORTANT:
        # Dataset images are stored as:
        # 0.tif, 1.tif, ..., 771.tif
        # The JSON original_name is NOT the actual filename.
        # --------------------------------------------------

        image_path = (
            f"{self.image_dir}/{sample['image_id']}.tif"
        )

        image = Image.open(image_path).convert("RGB")

        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = image_inputs["pixel_values"].squeeze(0)


        text_inputs = self.tokenizer(
            sample["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = text_inputs["input_ids"].squeeze(0)

        attention_mask = text_inputs["attention_mask"].squeeze(0)


        return {

            "image_id": sample["image_id"],

            "question_id": sample["question_id"],

            "pixel_values": pixel_values,

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "question_type": sample["question_type"],

            "question_type_label": sample["question_type_label"],

            "presence_label": sample["presence_label"],

            "comp_label": sample["comp_label"],

            "rural_urban_label": sample["rural_urban_label"],

            "count_value": sample["count_value"],

            "answer_label": sample["answer_label"],

            "answer": sample["answer"],

            "question": sample["question"]
        }
