def run_inference(
    image1_path,
    image2_path,
    question,
    question_type
):

    # -----------------------------
    # Preprocess images
    # -----------------------------

    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)

    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)


    # -----------------------------
    # Tokenize question
    # -----------------------------

    token_ids, attention_mask = tokenize_question(
        question,
        max_length=32
    )

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long
    ).unsqueeze(0).to(device)


    # -----------------------------
    # Question type
    # -----------------------------

    question_type_id = torch.tensor(
        [QUESTION_TYPE_TO_ID[question_type]],
        dtype=torch.long
    ).to(device)


    # -----------------------------
    # Model prediction
    # -----------------------------

    with torch.no_grad():

        outputs = cdvqa_model(
            image1=image1,
            image2=image2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_type_id=question_type_id
        )


    # -----------------------------
    # Select output head
    # -----------------------------

    if question_type in [
        "change_or_not",
        "increase_or_not",
        "decrease_or_not"
    ]:

        classes = ["no", "yes"]
        logits = outputs["binary"]


    elif question_type in [
        "smallest_change",
        "largest_change",
        "change_to_what"
    ]:

        classes = [
            "NVG_surface",
            "buildings",
            "low_vegetation",
            "trees",
            "water",
            "playgrounds"
        ]

        logits = outputs["category"]


    else:

        classes = [
            "0",
            "0_to_10",
            "10_to_20",
            "20_to_30",
            "30_to_40",
            "40_to_50",
            "50_to_60",
            "60_to_70",
            "70_to_80",
            "80_to_90",
            "90_to_100"
        ]

        logits = outputs["ratio"]


    # -----------------------------
    # Probability
    # -----------------------------

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction_id = probabilities.argmax(
        dim=1
    ).item()

    answer = classes[prediction_id]

    confidence = probabilities[
        0,
        prediction_id
    ].item()


    return answer, confidence