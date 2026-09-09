import os
import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, LayoutLMForTokenClassification


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "2-13aahana/layoutlm-document-analyzer"
TOKENIZER_NAME = "microsoft/layoutlm-base-uncased"


# ============================================================
# 2. LOAD TOKENIZER + MODEL + OCR
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

model = LayoutLMForTokenClassification.from_pretrained(
    MODEL_PATH
)

model.to("cpu")
model.eval()


ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)


print("✅ Everything loaded!")


# ============================================================
# 3. OCR → WORDS + ORIGINAL PIXEL BOXES
# ============================================================

def extract_words_and_boxes(image_path):

    result = ocr.predict(image_path)

    words = []
    boxes = []

    for res in result:

        data = res.json

        if callable(data):
            data = data()

        data = data["res"]

        texts = data["rec_texts"]
        text_boxes = data["rec_boxes"]

        for text, box in zip(texts, text_boxes):

            text = text.strip()

            if not text:
                continue

            x1, y1, x2, y2 = box

            words.append(text)

            boxes.append([
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])

    return words, boxes


# ============================================================
# 4. NORMALIZE BOX
# ============================================================

def normalize_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(1000 * x1 / width),
        int(1000 * y1 / height),
        int(1000 * x2 / width),
        int(1000 * y2 / height)
    ]


# ============================================================
# 5. NORMALIZE ALL BOXES
# ============================================================

def normalize_boxes(boxes, width, height):

    return [
        normalize_box(box, width, height)
        for box in boxes
    ]


# ============================================================
# 6. TOKENIZE DOCUMENT
# ============================================================

def tokenize_document(words):

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True
    )

    return encoding


# ============================================================
# 7. CREATE TOKEN-LEVEL BOXES
# ============================================================

def create_token_boxes(encoding, normalized_boxes):

    all_token_boxes = []

    for chunk_idx in range(
        len(encoding["input_ids"])
    ):

        word_ids = encoding.word_ids(
            batch_index=chunk_idx
        )

        token_boxes = []

        for word_id in word_ids:

            if word_id is None:

                # Special token / padding
                token_boxes.append(
                    [0, 0, 0, 0]
                )

            else:

                token_boxes.append(
                    normalized_boxes[word_id]
                )

        all_token_boxes.append(token_boxes)

    return all_token_boxes


# ============================================================
# 8. CONVERT TO PYTORCH TENSORS
# ============================================================

def create_model_inputs(
    encoding,
    token_boxes
):

    input_ids = torch.tensor(
        encoding["input_ids"],
        dtype=torch.long
    )

    attention_mask = torch.tensor(
        encoding["attention_mask"],
        dtype=torch.long
    )

    bbox = torch.tensor(
        token_boxes,
        dtype=torch.long
    )

    return input_ids, attention_mask, bbox


# ============================================================
# 9. RUN LAYOUTLM
# ============================================================

def predict(
    input_ids,
    attention_mask,
    bbox
):

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox
        )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )

    return predictions


# ============================================================
# 10. GET LABEL NAME
# ============================================================

def get_label_name(label_id):

    return model.config.id2label[
        int(label_id)
    ]


# ============================================================
# 11. CONVERT 7 BIO LABELS → 3 LABELS
# ============================================================

def simplify_label(label):

    if label == "O":
        return "O"

    if "-" in label:

        prefix, entity = label.split(
            "-",
            1
        )

        if entity in [
            "HEADER",
            "QUESTION",
            "ANSWER"
        ]:
            return entity

    return label


# ============================================================
# 12. CONVERT TOKEN PREDICTIONS → WORD PREDICTIONS
# ============================================================

def get_word_predictions(
    encoding,
    predictions,
    words
):

    word_predictions = {}

    for chunk_idx in range(
        len(predictions)
    ):

        word_ids = encoding.word_ids(
            batch_index=chunk_idx
        )

        for token_idx, word_id in enumerate(
            word_ids
        ):

            if word_id is None:
                continue

            # Keep the first prediction
            # for each word.
            if word_id not in word_predictions:

                label_id = predictions[
                    chunk_idx,
                    token_idx
                ]

                label = get_label_name(
                    label_id
                )

                label = simplify_label(
                    label
                )

                word_predictions[
                    word_id
                ] = label

    # Return predictions in word order
    results = []

    for word_id, word in enumerate(words):

        label = word_predictions.get(
            word_id,
            "O"
        )

        results.append({
            "word": word,
            "label": label
        })

    return results


# ============================================================
# 13. VISUALIZATION
# ============================================================

def visualize_predictions(
    image_path,
    words,
    boxes,
    predictions,
    output_path="prediction.png"
):

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            14
        )
    except:
        font = ImageFont.load_default()


    # Three beautiful separate colors
    colors = {
        "HEADER": "crimson",
        "QUESTION": "blue",
        "ANSWER": "green"
    }


    for word, box, prediction in zip(
        words,
        boxes,
        predictions
    ):

        label = prediction["label"]

        # Ignore O
        if label not in colors:
            continue

        x1, y1, x2, y2 = box

        color = colors[label]


        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=2
        )


        # Label at TOP-RIGHT
        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = (
            text_bbox[2] -
            text_bbox[0]
        )

        text_height = (
            text_bbox[3] -
            text_bbox[1]
        )

        text_x = x2 - text_width

        text_y = max(
            0,
            y1 - text_height - 2
        )


        draw.text(
            (text_x, text_y),
            label,
            fill=color,
            font=font
        )


    image.save(output_path)

    print(
        f"✅ Visualization saved to: {output_path}"
    )


# ============================================================
# 14. COMPLETE INFERENCE PIPELINE
# ============================================================

def run_inference(image_path):

    print("\n🔍 Processing:", image_path)


    # --------------------------------
    # Load image
    # --------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    print(
        "Image size:",
        width,
        "x",
        height
    )


    # --------------------------------
    # OCR
    # --------------------------------

    words, pixel_boxes = (
        extract_words_and_boxes(
            image_path
        )
    )

    print(
        "Number of words:",
        len(words)
    )


    # --------------------------------
    # Normalize boxes
    # --------------------------------

    normalized = normalize_boxes(
        pixel_boxes,
        width,
        height
    )


    # --------------------------------
    # Tokenization
    # --------------------------------

    encoding = tokenize_document(
        words
    )

    print(
        "Number of chunks:",
        len(encoding["input_ids"])
    )


    # --------------------------------
    # Token → box alignment
    # --------------------------------

    token_boxes = create_token_boxes(
        encoding,
        normalized
    )


    # --------------------------------
    # Convert to tensors
    # --------------------------------

    input_ids, attention_mask, bbox = (
        create_model_inputs(
            encoding,
            token_boxes
        )
    )


    print(
        "Input IDs:",
        input_ids.shape
    )

    print(
        "Attention mask:",
        attention_mask.shape
    )

    print(
        "BBox:",
        bbox.shape
    )


    # --------------------------------
    # LayoutLM prediction
    # --------------------------------

    predictions = predict(
        input_ids,
        attention_mask,
        bbox
    )


    print(
        "Logits / predictions generated!"
    )


    # --------------------------------
    # Word-level predictions
    # --------------------------------

    word_predictions = (
        get_word_predictions(
            encoding,
            predictions,
            words
        )
    )


    # --------------------------------
    # Visualization
    # --------------------------------

    visualize_predictions(
        image_path,
        words,
        pixel_boxes,
        word_predictions,
        output_path="prediction.png"
    )


    return {
        "words": words,
        "pixel_boxes": pixel_boxes,
        "predictions": word_predictions,
        "encoding": encoding,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox
    }


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    image_path = "test_image.png"

    result = run_inference(
        image_path
    )