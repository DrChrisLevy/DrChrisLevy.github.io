# ruff: noqa
import os
import shutil

import modal
from dotenv import load_dotenv
from modal import Image, build, enter

# ---------------------------------- SETUP BEGIN ----------------------------------#
ENV_FILE = ".env"  # path to local env file with wandb api key WANDB_API_KEY=<>
ds_name = "dair-ai/emotion"  # name of the Hugging Face dataset to use
ds_name_config = None  # for hugging face datasets that have multiple config instances. For example cardiffnlp/tweet_eval
checkpoint = "answerdotai/ModernBERT-base"  # name of the Hugging Face model to fine tune
batch_size = 32  # depends on GPU size and model size
GPU_SIZE = "A100"  # https://modal.com/docs/guide/gpu#specifying-gpu-type
num_train_epochs = 2
# define the labels for the dataset
id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
learning_rate = 5e-5  # learning rate for the optimizer
wandb_project = "hugging_face_training_jobs"  # name of the wandb project to use
pre_fix_name = ""  # optional prefix to the run name to differentiate it from other experiments
input_column = "text"  # Often commonly called "inputs". Depends on the dataset. This is the text input column name.
label_column = "label"  # This is the label column name.
train_split = "train"  # This is the train split name.
validation_split = "validation"  # This is the validation split name.
test_split = "test"
# This is a label that gets assigned to any example that is not classified by the model
# according to some probability threshold. It's only used for evaluation.
unknown_label_int = -1
unknown_label_str = "UNKNOWN"
# define the run name which is used in wandb and the model name when saving model checkpoints
run_name = f"{ds_name}-{ds_name_config}-{checkpoint}-{batch_size=}-{learning_rate=}-{num_train_epochs=}"


# This is the logic for tokenizing the input text. It's used in the dataset map function
# during training and evaluation.
def tokenizer_function_logic(example, tokenizer):
    return tokenizer(example[input_column], padding=True, truncation=True, return_tensors="pt", max_length=512)


# ---------------------------------- SETUP END----------------------------------#

if pre_fix_name:
    run_name = f"{pre_fix_name}-{run_name}"

label2id = {v: k for k, v in id2label.items()}
path_to_ds = os.path.join("/data", ds_name, ds_name_config if ds_name_config else "")

load_dotenv(ENV_FILE)
app = modal.App("trainer")

image = Image.debian_slim(python_version="3.11").run_commands(
    "apt-get update && apt-get install -y htop git",
    "pip3 install torch torchvision torchaudio",
    "pip install git+https://github.com/huggingface/transformers.git datasets accelerate scikit-learn python-dotenv wandb",
    # f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
    f'wandb login  {os.environ["WANDB_API_KEY"]}',
)

vol = modal.Volume.from_name("trainer-vol", create_if_missing=True)


@app.cls(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv(filename=ENV_FILE)],
    gpu=GPU_SIZE,
    timeout=60 * 60 * 10,
    container_idle_timeout=300,
)
class Trainer:
    def __init__(self, reload_ds=True):
        import torch

        self.reload_ds = reload_ds
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @build()
    @enter()
    def setup(self):
        from datasets import load_dataset, load_from_disk
        from transformers import (
            AutoTokenizer,
        )
        from transformers.utils import move_cache

        os.makedirs("/data", exist_ok=True)

        if not os.path.exists(path_to_ds) or self.reload_ds:
            try:
                # clean out the dataset folder
                shutil.rmtree(path_to_ds)
            except FileNotFoundError:
                pass
            self.ds = load_dataset(ds_name, ds_name_config)
            # Save dataset to disk
            self.ds.save_to_disk(path_to_ds)
        else:
            self.ds = load_from_disk(path_to_ds)

        move_cache()

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(self, example):
        return tokenizer_function_logic(example, self.tokenizer)

    def compute_metrics(self, pred):
        """
        To debug this function manually on some sample input in ipython you can create an input
        pred object like this:
        from transformers import EvalPrediction
        import numpy as np
        logits=[[-0.9559,  0.7553],
        [ 2.0987, -2.3868],
        [ 1.0143, -1.1551],
        [ 1.3666, -1.6074]]
        label_ids = [1, 0, 1, 0]
        pred = EvalPrediction(predictions=logits, label_ids=label_ids)
        """
        import numpy as np
        import torch
        from sklearn.metrics import f1_score

        # pred is EvalPrediction object i.e. from transformers import EvalPrediction
        logits = torch.tensor(pred.predictions)  # raw prediction logits from the model
        label_ids = pred.label_ids  # integer label ids classes
        labels = torch.tensor(label_ids).double().numpy()

        probs = logits.softmax(dim=-1).float().numpy()  # probabilities for each class
        preds = np.argmax(probs, axis=1)  # take the label with the highest probability
        f1_micro = f1_score(labels, preds, average="micro", zero_division=True)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=True)
        return {"f1_micro": f1_micro, "f1_macro": f1_macro}

    @modal.method()
    def train_model(self):
        import wandb
        import os
        from datasets import load_from_disk
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        os.environ["WANDB_PROJECT"] = wandb_project
        # Remove previous training model saves if exists for same run_name
        try:
            shutil.rmtree(os.path.join("/data", run_name))
        except FileNotFoundError:
            pass

        ds = load_from_disk(path_to_ds)
        # useful for debugging and quick training: Just downsample the dataset
        # for split in ds.keys():
        #     ds[split] = ds[split].shuffle(seed=42).select(range(1000))
        num_labels = len(id2label)
        tokenized_dataset = ds.map(self.tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # https://www.philschmid.de/getting-started-pytorch-2-0-transformers
        training_args = TrainingArguments(
            output_dir=os.path.join("/data", run_name),
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # PyTorch 2.0 specifics
            bf16=True,  # bfloat16 training
            torch_compile=True,  # optimizations
            optim="adamw_torch_fused",  # improved optimizer
            # logging & evaluation strategies
            logging_dir=os.path.join("/data", run_name, "logs"),
            logging_strategy="steps",
            logging_steps=200,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="wandb",
            run_name=run_name,
        )

        configuration = AutoConfig.from_pretrained(checkpoint)
        # these dropout values are noted here in case we want to tweak them in future
        # experiments.
        # configuration.hidden_dropout_prob = 0.1  # 0.1 is default
        # configuration.attention_probs_dropout_prob = 0.1  # 0.1 is default
        # configuration.classifier_dropout = None  # If None then defaults to hidden_dropout_prob
        configuration.id2label = id2label
        configuration.label2id = label2id
        configuration.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=configuration,
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset[train_split],
            eval_dataset=tokenized_dataset[validation_split],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        # Log the trainer script
        wandb.save(__file__)

    def load_model(self, check_point):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(check_point)
        tokenizer = AutoTokenizer.from_pretrained(check_point)
        return tokenizer, model

    @modal.method()
    def eval_model(self, check_point=None, split=validation_split):
        import os
        import numpy as np
        import pandas as pd
        import torch
        import wandb
        from datasets import load_from_disk
        from sklearn.metrics import classification_report

        os.environ["WANDB_PROJECT"] = wandb_project
        if check_point is None:
            # Will use most recent checkpoint by default. It may not be the "best" checkpoint/model.
            check_points = sorted(
                os.listdir(os.path.join("/data/", run_name)), key=lambda x: int(x.split("-")[1]) if x.startswith("checkpoint-") else 0
            )
            check_point = os.path.join("/data", run_name, check_points[-1])
        print(f"Evaluating Checkpoint {check_point}, split {split}")

        tokenizer, model = self.load_model(check_point)

        def tokenize_function(example):
            return tokenizer_function_logic(example, tokenizer)

        model.to(self.device)
        test_ds = load_from_disk(path_to_ds)[split]

        test_ds = test_ds.map(tokenize_function, batched=True, batch_size=batch_size)
        test_ds = test_ds.rename_column(label_column, "labels")

        def forward_pass(batch):
            """
            To debug this function manually on some sample input in ipython, take your dataset
            that has already been tokenized and create a batch object with this code:
            batch_size = 32
            test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            small_ds = test_ds.take(batch_size)
            batch = {k: torch.stack([example[k] for example in small_ds]) for k in small_ds[0].keys()}
            """
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in tokenizer.model_input_names}
            with torch.no_grad():
                output = model(**inputs)
                probs = torch.softmax(output.logits, dim=-1).round(decimals=2)
            return {"probs": probs.cpu().numpy()}

        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds = test_ds.map(forward_pass, batched=True, batch_size=batch_size)

        test_ds.set_format("pandas")
        df_test = test_ds[:]

        def pred_label(probs, threshold):
            # probs is a list of probabilities for one row of the dataframe
            probs = np.array(probs)
            max_prob = np.max(probs)
            predicted_class = np.argmax(probs)

            if max_prob < threshold:
                return unknown_label_int

            return predicted_class

        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print("-" * 60)
            print(f"{threshold=}\n")
            df_test[f"pred_labels"] = df_test["probs"].apply(pred_label, args=(threshold,))
            print(f"Coverage Rate:\n")
            predictions_mapped = df_test[f"pred_labels"].map({**id2label, unknown_label_int: unknown_label_str})
            print("Raw counts:")
            print(predictions_mapped.value_counts())
            print("\nProportions:\n")
            print(predictions_mapped.value_counts(normalize=True))
            print(f"\nConditional metrics (classification report on predicted subset != {unknown_label_str})")
            mask = df_test[f"pred_labels"] != unknown_label_int
            y = np.array([x for x in df_test[mask]["labels"].values])
            y_pred = np.array([x for x in df_test[mask][f"pred_labels"].values])
            report = classification_report(
                y,
                y_pred,
                target_names=[k for k, v in sorted(label2id.items(), key=lambda item: item[1])],
                digits=2,
                zero_division=0,
                output_dict=False,
                labels=sorted(list(range(len(id2label)))),
            )
            print(report)
            # --- Overall Accuracy (count "Unknown" as incorrect) ---
            # If ground truth is never 'unknown_label_int', then any prediction of "Unknown" is automatically wrong.
            overall_acc = (df_test["labels"] == df_test[f"pred_labels"]).mean()
            print(f"Overall Accuracy (counting '{unknown_label_str}' as wrong): {overall_acc:.2%}")
            print("-" * 60)

        print("Probability Distribution Max Probability Across All Classes")
        print(pd.DataFrame([max(x) for x in df_test["probs"]]).describe())
        # Ensure wandb is finished
        wandb.finish()


@app.local_entrypoint()
def main():
    trainer = Trainer(reload_ds=True)

    print(f"Training {run_name}")
    trainer.train_model.remote()

    # Will use most recent checkpoint by default. It may not be the "best" checkpoint/model.
    # Write the full path to the checkpoint here if you want to evaluate a specific model.
    # For example: check_point = '/data/run_name/checkpoint-1234/'
    check_point = None
    trainer.eval_model.remote(
        check_point=check_point,
        split=validation_split,
    )
