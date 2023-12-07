import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

import transformers
from datasets import load_metric
from transformers import AutoTokenizer
from transformers.models.mbart import MBartForConditionalGeneration
from model.custom_mbart_model import CustomMbartModel
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from custom_dataset.vi_ba_dataset import ViBaDataset


def get_metric(metric_, tokenizer_):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # pres and labels are pairs
        # filter out if labels == -100 or preds == -100
        preds = np.where(preds != -100, preds, tokenizer_.pad_token_id)
        decoded_preds = tokenizer_.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.

        labels = np.where(labels != -100, labels, tokenizer_.pad_token_id)
        decoded_labels = tokenizer_.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        # print(decoded_labels)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric_.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer_.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    return compute_metrics

def main():
    WORD_DROPOUT_RATIO = 0.15
    WORD_REPLACEMENT_RATIO = 0.15
    model_checkpoint = "pretrained/best_aligned"
    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = CustomMbartModel.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer.get_vocab()))
    model.set_augment_config(word_dropout_ratio=WORD_DROPOUT_RATIO,
                             word_replacement_ratio=WORD_REPLACEMENT_RATIO)
    compute_metric_func = get_metric(metric, tokenizer)

    train_dataset, valid_dataset, test_dataset = ViBaDataset.get_datasets(data_folder="data/all",
                                                                          tokenizer_path=model_checkpoint)

    batch_size = 4

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer_args = Seq2SeqTrainingArguments(
        "/content/checkpoint/viba_bart-finetuned",
        metric_for_best_model="bleu",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        logging_steps=100,
        logging_first_step=True,
        logging_dir="logging/viba_bart-finetuned",
        eval_steps=100
    )

    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric_func
    )
    print(trainer.evaluate(eval_dataset=valid_dataset, num_beams=5, max_length=512))

    trainer.train()

    print(trainer.evaluate(eval_dataset=test_dataset, num_beams=5, max_length=512))
    trainer.save_model("checkpoint/best_4_split_num")


if __name__ == "__main__":
    main()
