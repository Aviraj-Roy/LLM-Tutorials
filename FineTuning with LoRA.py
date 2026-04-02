from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainin
from transformers import DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "distilgpt2"

dataset = load_dataset("imdb", split="train[:1%]")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

ds = dataset.map(tok, batched=True, remove_columns=["text"])

base = AutoModelForCausalLM.from_pretrained(model_name)

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"],
)

model = get_peft_model(base, lora)

args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
)

trainer.train()