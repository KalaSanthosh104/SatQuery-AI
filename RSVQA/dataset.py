
import json
from pathlib import Path
from torch.utils.data import Dataset


class RSVQADataset(Dataset):

    def __init__(self, dataset_root, split="train"):
        self.dataset_root = Path(dataset_root)
        self.split = split

        # ---------------------------------------------------------
        # Paths
        # ---------------------------------------------------------
        image_dir = self.dataset_root / "Images_LR" / "Images_LR"

        questions_path = (
            self.dataset_root /
            f"LR_split_{split}_questions.json"
        )

        answers_path = (
            self.dataset_root /
            f"LR_split_{split}_answers.json"
        )

        images_path = (
            self.dataset_root /
            f"LR_split_{split}_images.json"
        )

        # ---------------------------------------------------------
        # Load JSON files
        # ---------------------------------------------------------
        with open(questions_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        with open(answers_path, "r", encoding="utf-8") as f:
            answers_data = json.load(f)

        with open(images_path, "r", encoding="utf-8") as f:
            images_data = json.load(f)

        # ---------------------------------------------------------
        # Active records only
        # ---------------------------------------------------------
        questions = [
            q for q in questions_data["questions"]
            if q.get("active", False)
        ]

        answers = [
            a for a in answers_data["answers"]
            if a.get("active", False)
        ]

        images = [
            i for i in images_data["images"]
            if i.get("active", False)
        ]

        # ---------------------------------------------------------
        # Dictionaries
        # ---------------------------------------------------------
        answer_dict = {
            a["id"]: a["answer"]
            for a in answers
            if "answer" in a
        }

        image_dict = {
            i["id"]: i
            for i in images
        }

        # ---------------------------------------------------------
        # Answer vocabulary
        #
        # Only used to provide answer_label.
        # It NEVER filters samples.
        # ---------------------------------------------------------
        vocab_path = (
            self.dataset_root.parent /
            "project" /
            "answer_vocab.json"
        )

        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                answer_vocab = json.load(f)

            # Support either:
            # {"answer": id}
            # or
            # {"answers": {"answer": id}}
            if "answers" in answer_vocab:
                answer_to_id = answer_vocab["answers"]
            else:
                answer_to_id = answer_vocab
        else:
            answer_to_id = {}

        # ---------------------------------------------------------
        # Question type mapping
        # ---------------------------------------------------------
        question_type_to_id = {
            "presence": 0,
            "comp": 1,
            "rural_urban": 2,
            "count": 3
        }

        self.samples = []

        skipped = {
            "no_answer_id": 0,
            "missing_answer": 0,
            "missing_image": 0,
            "unknown_type": 0,
            "invalid_count": 0
        }

        unseen_answers = 0

        # ---------------------------------------------------------
        # Build samples
        # ---------------------------------------------------------
        for q in questions:

            question_id = q["id"]
            image_id = q["img_id"]
            question_type = q["type"]
            question_text = q["question"]

            # -------------------------
            # Answer ID
            # -------------------------
            answers_ids = q.get("answers_ids", [])

            if not answers_ids:
                skipped["no_answer_id"] += 1
                continue

            answer_id = answers_ids[0]

            if answer_id not in answer_dict:
                skipped["missing_answer"] += 1
                continue

            answer = answer_dict[answer_id]

            # -------------------------
            # Image
            # -------------------------
            if image_id not in image_dict:
                skipped["missing_image"] += 1
                continue

            image_path = image_dir / f"{image_id}.tif"

            # -------------------------
            # Question type
            # -------------------------
            if question_type not in question_type_to_id:
                skipped["unknown_type"] += 1
                continue

            question_type_label = question_type_to_id[question_type]

            # -------------------------
            # Default task labels
            # -------------------------
            presence_label = -1
            comp_label = -1
            rural_urban_label = -1
            count_value = -1.0

            # -------------------------
            # Presence
            # -------------------------
            if question_type == "presence":

                answer_lower = answer.lower().strip()

                if answer_lower == "yes":
                    presence_label = 1
                elif answer_lower == "no":
                    presence_label = 0

            # -------------------------
            # Composition
            # -------------------------
            elif question_type == "comp":

                answer_lower = answer.lower().strip()

                if answer_lower == "yes":
                    comp_label = 1
                elif answer_lower == "no":
                    comp_label = 0

            # -------------------------
            # Rural / Urban
            # -------------------------
            elif question_type == "rural_urban":

                answer_lower = answer.lower().strip()

                if answer_lower == "rural":
                    rural_urban_label = 0
                elif answer_lower == "urban":
                    rural_urban_label = 1

            # -------------------------
            # Count
            # -------------------------
            elif question_type == "count":

                try:
                    count_value = float(answer)
                except (ValueError, TypeError):
                    skipped["invalid_count"] += 1
                    continue

            # -------------------------
            # Answer vocabulary label
            # -------------------------
            if answer in answer_to_id:
                answer_label = answer_to_id[answer]
            else:
                answer_label = -1
                unseen_answers += 1

            # -------------------------
            # Store sample
            # -------------------------
            self.samples.append({
                "image_id": image_id,
                "image_path": str(image_path),

                "question_id": question_id,
                "question": question_text,

                "question_type": question_type,
                "question_type_label": question_type_label,

                "answer_id": answer_id,
                "answer": answer,
                "answer_label": answer_label,

                "presence_label": presence_label,
                "comp_label": comp_label,
                "rural_urban_label": rural_urban_label,
                "count_value": count_value
            })

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------
        print(f"\n{split.upper()} DATASET")
        print("-" * 50)
        print(f"Active questions : {len(questions)}")
        print(f"Final samples    : {len(self.samples)}")
        print(f"Skipped          : {len(questions) - len(self.samples)}")
        print(f"Unseen answers   : {unseen_answers}")

        print("\nSkip details:")
        for key, value in skipped.items():
            print(f"  {key:18s}: {value}")

        print("-" * 50)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
