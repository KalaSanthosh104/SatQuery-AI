# RSVQA Model - SatQuery Integration

## Files
- `best_count_finetuned_rsvqa.pt` - final trained checkpoint
- `model.py` - RSVQA model architecture
- `data_pipeline.py` - original dataset/input preprocessing code
- `dataset.py` - original dataset class
- `answer_vocab.json` - answer vocabulary used in the project
- `inference.py` - deployment/inference wrapper
- `requirements.txt` - Python dependencies

## Pretrained models
- ViT: `google/vit-base-patch16-224-in21k`
- BERT: `bert-base-uncased`

## Supported question types
1. Presence: yes/no
2. Composition: yes/no comparison
3. Rural/Urban: rural or urban
4. Count: numerical prediction

## Run
Put `inference.py`, `model.py`, and the checkpoint in the same folder.

Install:
`pip install -r requirements.txt`

Run:
`python inference.py`

The script asks for:
- checkpoint path
- image path
- question

For backend integration, import:
`from inference import RSVQAInference`

Then:
`predictor = RSVQAInference("best_count_finetuned_rsvqa.pt")`
`result = predictor.predict(image_path, question)`
