import transformers
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EncoderDecoderModel
from model.custom_bert2bert_model import CustomBERT2BERTModel

import numpy as np

from custom_dataset.vi_ba_aligned_dataset import ViBaDataset


def get_metric(metric_, tokenizer_):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer_.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer_.pad_token_id)
        decoded_labels = tokenizer_.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
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

    model_checkpoint = "pretrained/bert2bert_1"
    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trucation=True)
    model = CustomBERT2BERTModel.from_encoder_decoder_pretrained(model_checkpoint, model_checkpoint)
    # model = CustomBERT2BERTModel.from_encoder_decoder_pretrained(model_checkpoint, model_checkpoint)
    # model = CustomBERT2BERTModel.from_pretrained("/content/drive/MyDrive/Thac Si/Thesis/BaViBARTModel/checkpoint/bert2bert_best_1")

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.mask_token_id = tokenizer.mask_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    model.set_augment_config(word_dropout_ratio=WORD_DROPOUT_RATIO,
                             word_replacement_ratio=WORD_REPLACEMENT_RATIO)

    compute_metric_func = get_metric(metric, tokenizer)

    train_dataset, valid_dataset, test_dataset = ViBaDataset.get_datasets(data_folder="data/new_all",
                                                                          tokenizer_path=model_checkpoint)

    batch_size = 4

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=256)

    trainer_args = Seq2SeqTrainingArguments(
        "/content/checkpoint/bert2bert_1",
        metric_for_best_model="bleu",
        evaluation_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        logging_steps=1000,
        logging_first_step=True,
        logging_dir="/content/logging/viba_bert2bert-finetuned",
        eval_steps=500,
        load_best_model_at_end=True,
        generation_num_beams=5
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
    result = trainer.evaluate(eval_dataset=valid_dataset, num_beams=5, max_length=256)
    print(result)
    trainer.train()

    result = trainer.evaluate(test_dataset, num_beams=5, max_length=256)
    print(result)
    trainer.save_model("/content/checkpoint/bert2bert_best_1")


if __name__ == "__main__":
    main()
