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
        total_samples = 0

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

            # Accuracy 계산 (시퀀스 단위)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            for pred_text, label_text in zip(pred_texts, label_texts):
                if pred_text == label_text:
                    total_correct += 1
                total_samples += 1

            if (batch_idx + 1) % 10 == 0:
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                )

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(total_correct / total_samples)
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
    total_samples = 0

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
                if pred_text == label_text:
                    total_correct += 1
                total_samples += 1

    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_samples
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
    total_samples = 0
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
                if pred_text == label_text:
                    total_correct += 1
                total_samples += 1

    accuracy = total_correct / total_samples
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
    total_samples = 0

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
                if pred_text == ref_text:
                    total_correct += 1
                total_samples += 1

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"\n[{phase} Full Evaluation]")
    print(f"Loss: {avg_loss:.4f}, Seq-Level Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

from torch.utils.tensorboard import SummaryWriter
import os

# TensorBoard SummaryWriter
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)
run_name = "large_model"
writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

# 학습 루프
def train_with_full_and_random_evaluation(model, tokenizer, train_loader, val_loader, test_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    os.makedirs("./saved_models", exist_ok=True)  # 모델 저장 디렉토리 생성

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

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

            # Accuracy 계산 (시퀀스 단위)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            for pred_text, label_text in zip(pred_texts, label_texts):
                if pred_text == label_text:
                    total_correct += 1
                total_samples += 1

            seq_accuracy = total_correct / total_samples if total_samples > 0 else 0

            # TensorBoard에 학습 손실 및 정확도 기록 (Batch 단위)
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Train/Seq_Accuracy", seq_accuracy, epoch * len(train_loader) + batch_idx)

            # Iteration마다 출력
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Seq Accuracy: {seq_accuracy:.4f}"
                )

                # Train, Validation, Test 랜덤 배치 평가
                evaluate_random_batch(model, tokenizer, train_loader, phase="Train")
                evaluate_random_batch(model, tokenizer, val_loader, phase="Validation")
                evaluate_random_batch(model, tokenizer, test_loader, phase="Test")

        # Epoch 종료 후 평균 손실 및 정확도 기록
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_correct / total_samples if total_samples > 0 else 0
        writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/Train_Seq_Accuracy", avg_train_accuracy, epoch)

        print(f"\nEpoch {epoch + 1}/{epochs} completed! Average Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, Train Seq Accuracy: {avg_train_accuracy:.4f}")

        # Full evaluation for Validation and Test sets
        val_loss, val_accuracy = evaluate_full_loader(model, tokenizer, val_loader, phase="Validation")
        test_loss, test_accuracy = evaluate_full_loader(model, tokenizer, test_loader, phase="Test")

        # TensorBoard에 Validation 및 Test 손실 및 정확도 기록
        writer.add_scalar("Epoch/Validation_Loss", val_loss, epoch)
        writer.add_scalar("Epoch/Validation_Seq_Accuracy", val_accuracy, epoch)
        writer.add_scalar("Epoch/Test_Loss", test_loss, epoch)
        writer.add_scalar("Epoch/Test_Seq_Accuracy", test_accuracy, epoch)

        print(f"Validation Loss: {val_loss:.4f}, Validation Seq Accuracy: {val_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Seq Accuracy: {test_accuracy:.4f}")

        # 모델 저장 (Validation Accuracy 포함)
        model_save_path = f"./saved_models/bart_large_epoch_{epoch + 1}_val_acc_{val_accuracy:.4f}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


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
