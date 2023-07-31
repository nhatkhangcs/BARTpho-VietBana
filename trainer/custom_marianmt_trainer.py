import transformers
from datasets import load_metric
from transformers import AutoTokenizer
from model.custom_marian_model import CustomMarianModel
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from custom_dataset.vi_ba_aligned_dataset import ViBaDataset


def get_metric(metric_, tokenizer_):
    with tokenizer_.as_target_tokenizer():
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
    WORD_REPLACEMENT_RATIO = 0.00
    model_checkpoint = "Helsinki-NLP/opus-mt-vi-en"
    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained("pretrained/marian")
    # model = CustomMbartModel.from_pretrained("/content/drive/MyDrive/Thac Si/Thesis/BaViBARTModel/checkpoint/best")
    model = CustomMarianModel.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer.get_vocab()))
    model.set_augment_config(word_dropout_ratio=WORD_DROPOUT_RATIO,
                             word_replacement_ratio=WORD_REPLACEMENT_RATIO)
    compute_metric_func = get_metric(metric, tokenizer)

    train_dataset, valid_dataset, test_dataset = ViBaDataset.get_datasets(data_folder="data/new_all",
                                                                          tokenizer_path=model_checkpoint)

    batch_size = 4

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer_args = Seq2SeqTrainingArguments(
        "/content/checkpoint/viba_marian-finetuned",
        metric_for_best_model="bleu",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        logging_steps=1000,
        logging_first_step=True,
        load_best_model_at_end=True,
        logging_dir="logging/viba_marian-finetuned",
        eval_steps=500
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
    print(trainer.evaluate(eval_dataset=valid_dataset, num_beams=5, max_length=256))

    trainer.train()

    print(trainer.evaluate(test_dataset, num_beams=5, max_length=256))
    trainer.save_model("checkpoint/best_aligned_marian")


if __name__ == "__main__":
    main()
