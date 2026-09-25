import torch
from PIL import Image

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BEST_CDVQA_PATH = (
    "/kaggle/input/models/"
    "santuu112/cdvqa-best-model/"
    "pytorch/default/1/"
    "best_cdvqa_model.pth"
)

cdvqa_model = CDVQAModel(
    vocab_size=len(word_to_id),
    embedding_dim=128,
    hidden_dim=128,
    num_question_types=8,
    num_categories=6,
    num_ratios=11
).to(device)


checkpoint = torch.load(
    BEST_CDVQA_PATH,
    map_location=device
)

cdvqa_model.load_state_dict(
    checkpoint["model_state_dict"]
)

cdvqa_model.eval()

print("CDVQA model loaded successfully")