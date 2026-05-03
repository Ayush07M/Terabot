!pip install transformers datasets tensorflow sentencepiece accelerate

!pip install --upgrade datasets huggingface_hub
from huggingface_hub import list_datasets

datasets_list = list_datasets()

mental_health_datasets = []
for d in datasets_list:
    if "mental" in d.id.lower() or "dialog" in d.id.lower():
        mental_health_datasets.append(d)

print(mental_health_datasets)

from datasets import load_dataset

# Load both datasets
chatbot_dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
counseling_dataset = load_dataset("Amod/mental_health_counseling_conversations")

# Print dataset structures
print(chatbot_dataset)
print(counseling_dataset)

import pandas as pd
chatbot_df = pd.read_csv("mental_health_chatbot.csv")
counseling_df = pd.read_csv("mental_health_counseling.csv")
chatbot_df["Context"] = chatbot_df["Context"].str.replace(r"<HUMAN>\s*", "", regex=True)
combined_df = pd.concat([chatbot_df, counseling_df], ignore_index=True)
# Save the final cleaned dataset
combined_df.to_csv("combined_dataset.csv", index=False)

print(combined_df.head())

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

contexts = combined_df["Context"].values
responses = combined_df["Response"].values

contexts = [str(c) for c in contexts]
responses = [str(r) for r in responses]

tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(contexts + responses)

context_sequences = tokenizer.texts_to_sequences(contexts)
response_sequences = tokenizer.texts_to_sequences(responses)

max_length = max(max(len(seq) for seq in context_sequences), max(len(seq) for seq in response_sequences))
context_sequences = pad_sequences(context_sequences, maxlen=max_length, padding="post", truncating="post")
response_sequences = pad_sequences(response_sequences, maxlen=max_length, padding="post", truncating="post")


vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

print(f"Vocabulary Size: {vocab_size}")
print(f"Sample Context Sequence: {context_sequences[0]}")
print(f"Sample Response Sequence: {response_sequences[0]}")

word_freq = tokenizer.word_counts
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
print(sorted_words)

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

from datasets import Dataset
from transformers import BlenderbotTokenizer

data = {
    "context": contexts,
    "response": responses
}

# Step 2: Create a Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Step 3: Load Blenderbot tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

# Step 4: Define tokenization function
def tokenize_function(example):
    model_inputs = tokenizer(example["context"], truncation=True, padding='max_length', max_length=128)
    labels = tokenizer(example["response"], truncation=True, padding='max_length', max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 5: Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import BlenderbotForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Define training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./blenderbot-mentalhealth",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # double it if possible
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    optim="adamw_torch",  # explicitly set AdamW optimizer
    lr_scheduler_type="cosine",  # change scheduler!
    gradient_checkpointing=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("path/to/your/credentials.json", scope)

client = gspread.authorize(creds)

sheet = client.open("MentalHealthConversations").sheet1  # assumes you created a sheet with this name

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_path = "./results"
tokenizer = BlenderbotTokenizer.from_pretrained(model_path)
model = BlenderbotForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("🤖 Hello! I'm your AI therapist. Type 'exit' to end the chat.")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("🤖 Take care! We'll talk again soon.")
        break

    # Append user input to history
    chat_history.append(user_input)


    context = " ".join(chat_history[-5:])

    inputs = tokenizer(context, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=150)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    print("Bot:", response)

    chat_history.append(response)

    # Save interaction to Google Sheet
sheet.append_row([user_input, response])