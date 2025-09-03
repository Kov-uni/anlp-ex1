import argparse
import torch
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_predict", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="./final-mrpc-model2")
    args = parser.parse_args()

    '''    
    In order to use wandb:
    run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
    #wandb.init(project="mrpc-paraphrase-detection", name=run_name)
    '''

    dataset = load_dataset('glue', 'mrpc')
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'],
                         truncation=True, padding=False)

    if args.max_train_samples != -1:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        dataset["test"] = dataset["test"].select(range(args.max_predict_samples))

    tokenized = dataset.map(preprocess, batched=True)

    accuracy_metric = load_metric("glue", "mrpc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictionsList = np.argmax(logits, axis=1)
        return accuracy_metric.compute(predictions=predictionsList, references=labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.do_train:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        training_args = TrainingArguments(
            output_dir="./results",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=30,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.06,
            ##report_to=["wandb"],
            report_to=[],
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(args.model_path)

        model.eval()
        eval_results = trainer.evaluate(eval_dataset=tokenized["validation"])
        val_acc = eval_results["eval_accuracy"]

        with open("./res.txt", "a") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n")

    if args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.eval()

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
        )

        predictions = trainer.predict(tokenized["test"])
        pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)

        raw_test = dataset["test"]
        sentences1 = raw_test["sentence1"]
        sentences2 = raw_test["sentence2"]

        with open("./predictions.txt", "w", encoding="utf-8") as f:
            for s1, s2, label in zip(sentences1, sentences2, pred_labels.tolist()):
                f.write(f"{s1}###{s2}###{label}\n")

if __name__ == "__main__":
    main()
