from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import ImageDraw, ImageFont


class LabelSet:
    theivaprakasham = {
        "labels": [
            "O",
            "B-ABN",
            "B-BILLER",
            "B-BILLER_ADDRESS",
            "B-BILLER_POST_CODE",
            "B-DUE_DATE",
            "B-GST",
            "B-INVOICE_DATE",
            "B-INVOICE_NUMBER",
            "B-SUBTOTAL",
            "B-TOTAL",
            "I-BILLER_ADDRESS",
        ],
        "colors": {
            "B-ABN": "blue",
            "B-BILLER": "blue",
            "B-BILLER_ADDRESS": "green",
            "B-BILLER_POST_CODE": "orange",
            "B-DUE_DATE": "blue",
            "B-GST": "green",
            "B-INVOICE_DATE": "violet",
            "B-INVOICE_NUMBER": "orange",
            "B-SUBTOTAL": "green",
            "B-TOTAL": "blue",
            "I-BILLER_ADDRESS": "blue",
            "O": "orange",
        },
    }
    nielsr_layoutlmv3_finetuned_funsd = {
        "labels": [
            "O",
            "B-HEADER",
            "I-HEADER",
            "B-QUESTION",
            "I-QUESTION",
            "B-ANSWER",
            "I-ANSWER",
        ],
        "colors": {
            "B-HEADER": "blue",
            "I-HEADER": "blue",
            "B-QUESTION'": "green",
            "I-QUESTIONE": "orange",
            "B-ANSWER": "violet",
            "I-ANSWER": "violet",
            "O": "orange",
        },
    }


class LayoutLvm3:
    def __init__(self, model_name="nielsr/layoutlmv3-finetuned-funsd"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        if model_name == "Theivaprakasham/layoutlmv3-finetuned-invoice":
            self.labels = LabelSet.theivaprakasham["labels"]
            self.label2color = LabelSet.theivaprakasham["colors"]

        elif model_name == "nielsr/layoutlmv3-finetuned-funsd":
            self.labels = LabelSet.nielsr_layoutlmv3_finetuned_funsd["labels"]
            self.label2color = LabelSet.nielsr_layoutlmv3_finetuned_funsd["colors"]
        else:
            self.labels = [
                self.model.config.id2label[i]
                for i in sorted(self.model.config.id2label)
            ]
            self.label2color = {label: "black" for label in self.labels}

    def infer(self, image, words, bboxs):
        max_length = 512
        all_predictions = []
        all_token_boxes = []

        for i in range(0, len(words), max_length):
            chunk_words = words[i : i + max_length]
            chunk_boxes = bboxs[i : i + max_length]

            encoding = self.processor(
                image,
                chunk_words,
                boxes=chunk_boxes,
                return_tensors="pt",
                truncation=True,
            )
            outputs = self.model(**encoding)

            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding.bbox.squeeze().tolist()

            all_predictions.extend(predictions)
            all_token_boxes.extend(token_boxes)

        return all_predictions, all_token_boxes

    def draw(self, image, words, boxes, predictions, box_format="xywh"):
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        for word, box, pred in zip(words, boxes, predictions):
            if pred >= len(self.labels):
                continue

            label = self.labels[pred]
            if label == "O":
                continue

            color = self.label2color.get(label, "black")

            if box_format == "xywh":
                x, y, w, h = box
                box = [x, y, x + w, y + h]
            elif box_format == "xyxy":
                pass
            else:
                raise ValueError("Unsupported box format. Use 'xywh' or 'xyxy'.")

            x0, y0, x1, y1 = box
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0 + 2, y0 - 12), label, fill=color, font=font)

        return image
