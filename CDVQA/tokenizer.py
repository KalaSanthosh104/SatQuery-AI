def tokenize_question(question, max_length=32):

    words = question.lower().split()

    token_ids = [
        word_to_id.get(
            word,
            word_to_id["<UNK>"]
        )
        for word in words
    ]

    # Truncate
    token_ids = token_ids[:max_length]

    # Attention mask
    attention_mask = [1] * len(token_ids)

    # Padding
    while len(token_ids) < max_length:
        token_ids.append(word_to_id["<PAD>"])
        attention_mask.append(0)

    return token_ids, attention_mask