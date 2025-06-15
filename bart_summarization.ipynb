# STEP 1: Install Required Libraries
!pip install transformers[torch] datasets accelerate rouge_score --quiet
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --quiet

# STEP 2: Check GPU Availability
import torch

if torch.cuda.is_available():
    print(f"\u2705 GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("\u26A0\uFE0F GPU not available. Training may be slow.")

# STEP 3: Load CNN/DailyMail Dataset and Tokenizer
from datasets import load_dataset
from transformers import BartTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)

dataset = load_dataset("cnn_dailymail", '3.0.0')

# STEP 4: Preprocess Data
def preprocess_function(examples):
    model_inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

# STEP 5: Create DataLoaders (Small Subset)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=4)
eval_dataloader = DataLoader(small_eval_dataset, collate_fn=data_collator, batch_size=4)

# STEP 6: Load Model and Define Trainer
from transformers import BartForConditionalGeneration, Trainer, TrainingArguments

model = BartForConditionalGeneration.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# STEP 7: Train the Model
trainer.train()

# STEP 8: Define Summary Generator
def generate_summaries(texts):
    inputs = tokenizer(texts, max_length=1024, truncation=True, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.batch_decode(summaries, skip_special_tokens=True)

# STEP 9: Evaluate Using ROUGE
from datasets import load_metric
rouge_metric = load_metric("rouge")

sample_texts = [ex["article"] for ex in small_eval_dataset.select(range(10))]
reference_summaries = [ex["highlights"] for ex in small_eval_dataset.select(range(10))]

predicted_summaries = generate_summaries(sample_texts)

results = rouge_metric.compute(predictions=predicted_summaries, references=reference_summaries, use_stemmer=True)

for key in ['rouge1', 'rouge2', 'rougeL']:
    score = results[key].mid
    print(f"{key.upper()} - Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1: {score.fmeasure:.4f}")
