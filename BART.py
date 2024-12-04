from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset 정의
class DateExtractionDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenization
        input_enc = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0),
        }

# 데이터 로드
def load_data(file_path, tokenizer, max_length=128, batch_size=16):
    input_texts = []
    target_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            input_text, target_text = line.strip().split(":____GT:")
            input_texts.append(input_text.strip())
            target_texts.append(target_text.strip())

    dataset = DateExtractionDataset(input_texts, target_texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 모델 및 토크나이저 로드
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# 손실 함수 및 옵티마이저
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 학습 루프
def train_model(model, tokenizer, train_loader, val_loader, epochs=3):
    model.train()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_chars = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy 계산 (글자 단위)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            for pred_text, label_text in zip(pred_texts, label_texts):
                total_correct += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)

            if (batch_idx + 1) % 10 == 0:
                char_accuracy = total_correct / total_chars if total_chars > 0 else 0
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Char Accuracy: {char_accuracy:.4f}"
                )

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(total_correct / total_chars)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}")

        # Validation
        val_loss, val_accuracy = validate_model(model, tokenizer, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    # Loss 및 Accuracy 그래프
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# 검증 루프
def validate_model(model, tokenizer, val_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            for pred_text, label_text in zip(pred_texts, label_texts):
                total_correct += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)

    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_chars
    return val_loss, val_accuracy

# 결과 그래프 출력
def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

# 테스트 평가
def evaluate_test_set(model, tokenizer, test_loader):
    model.eval()
    total_correct = 0
    total_chars = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(pred_texts)
            references.extend(label_texts)

            for pred_text, label_text in zip(pred_texts, label_texts):
                total_correct += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)

    accuracy = total_correct / total_chars
    print(f"Test Accuracy: {accuracy}")
    return predictions, references
# 랜덤 배치 평가
def evaluate_random_batch(model, tokenizer, loader, phase):
    model.eval()
    random_batch = random.choice(list(loader))

    input_ids = random_batch["input_ids"].to(device)
    attention_mask = random_batch["attention_mask"].to(device)
    labels = random_batch["labels"].to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(f"\n[{phase} Random Batch Example]")
        for inp, pred, ref in zip(random_batch["input_ids"][:2], pred_texts[:2], ref_texts[:2]):
            print(f"Input: {tokenizer.decode(inp, skip_special_tokens=True)}")
            print(f"Predicted: {pred}")
            print(f"Reference: {ref}\n")

# 전체 데이터셋 평가
def evaluate_full_loader(model, tokenizer, loader, phase):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred_text, ref_text in zip(pred_texts, ref_texts):
                total_correct += sum(p == r for p, r in zip(pred_text, ref_text))
                total_chars += len(ref_text)

    avg_loss = total_loss / len(loader)
    char_accuracy = total_correct / total_chars if total_chars > 0 else 0

    print(f"\n[{phase} Full Evaluation]")
    print(f"Loss: {avg_loss:.4f}, Char-Level Accuracy: {char_accuracy:.4f}")
    return avg_loss, char_accuracy

# 학습 루프
def train_with_full_and_random_evaluation(model, tokenizer, train_loader, val_loader, test_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy 계산 (글자 단위)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            total_correct = 0
            total_chars = 0

            for pred_text, label_text in zip(pred_texts, label_texts):
                total_correct += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)

            char_accuracy = total_correct / total_chars if total_chars > 0 else 0

            # Iteration마다 출력
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Char Accuracy: {char_accuracy:.4f}"
                )

                # Train, Validation, Test 랜덤 배치 평가
                evaluate_random_batch(model, tokenizer, train_loader, phase="Train")
                evaluate_random_batch(model, tokenizer, val_loader, phase="Validation")
                evaluate_random_batch(model, tokenizer, test_loader, phase="Test")

        print(f"\nEpoch {epoch + 1}/{epochs} completed! Average Train Loss: {total_loss / len(train_loader):.4f}")

        # Full evaluation at the end of the epoch
        evaluate_full_loader(model, tokenizer, train_loader, phase="Train")
        evaluate_full_loader(model, tokenizer, val_loader, phase="Validation")
        evaluate_full_loader(model, tokenizer, test_loader, phase="Test")


# 데이터 경로
train_file = "./data/train/train.txt"
valid_file = "./data/valid/valid.txt"
test_file = "./data/test/test.txt"

# 데이터 로더 준비
train_loader = load_data(train_file, tokenizer, max_length=128, batch_size=16)
val_loader = load_data(valid_file, tokenizer, max_length=128, batch_size=16)
test_loader = load_data(test_file, tokenizer, max_length=128, batch_size=16)

train_with_full_and_random_evaluation(model, tokenizer, train_loader, val_loader, test_loader, epochs=3)

# 학습 실행
# train_model(model, tokenizer, train_loader, val_loader, epochs=5)

# 테스트 데이터 성능 평가
predictions, references = evaluate_test_set(model, tokenizer, test_loader)

# 결과 출력
for pred, ref in zip(predictions[:10], references[:10]):
    print(f"Predicted: {pred} | Reference: {ref}")
