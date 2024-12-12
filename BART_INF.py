from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os

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

# 모델 로드 함수
def load_model_from_checkpoint(model_name, checkpoint_path, tokenizer_name):
    print(f"Loading model from {checkpoint_path}")
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    model = BartForConditionalGeneration.from_pretrained(tokenizer_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# Inference 및 결과 저장 함수
def evaluate_and_save_results(model, tokenizer, test_loader, output_file="inference_results.txt"):
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions = []
    references = []
    inputs = []

    # 저장할 파일 초기화
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Input\tPredicted\tReference\n")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Inference
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            predictions.extend(pred_texts)
            references.extend(ref_texts)
            inputs.extend(input_texts)

            # Accuracy 계산 및 결과 저장
            with open(output_file, "a", encoding="utf-8") as f:
                for inp, pred, ref in zip(input_texts, pred_texts, ref_texts):
                    f.write(f"{inp}\t{pred}\t{ref}\n")
                    if pred == ref:
                        total_correct += 1
                    total_samples += 1

    # Accuracy 계산
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")
    return inputs, predictions, references, accuracy

# 체크포인트 파일 경로 및 설정
## BART - large 모델
checkpoint_path = "./BART_large_val_acc_0.9369.pth"
test_file = "./test.txt"
output_file = "./inference_results.txt"
model_name = "facebook/bart-large"

# 체크포인트 파일 경로 및 설정
## BART - base 모델
checkpoint_path = "./BART_base_val_acc_0.9406.pth"
test_file = "./test.txt"
output_file = "./inference_results.txt"
model_name = "facebook/bart-base"

# 데이터 로더 준비
tokenizer = BartTokenizer.from_pretrained(model_name)
test_loader = load_data(test_file, tokenizer, max_length=128, batch_size=16)

# 모델 로드 및 테스트
model, tokenizer = load_model_from_checkpoint(
    model_name = model_name,
    checkpoint_path = checkpoint_path,
    tokenizer_name = model_name
)

# 테스트 데이터 평가 및 결과 저장
inputs, predictions, references, accuracy = evaluate_and_save_results(model, tokenizer, test_loader, output_file)

# 결과 출력
print("\nSample Results:")
for inp, pred, ref in zip(inputs[:10], predictions[:10], references[:10]):
    print(f"Input: {inp}")
    print(f"Predicted: {pred}")
    print(f"Reference: {ref}")
    print("-" * 50)

# 저장된 결과 확인
print(f"Inference results saved to: {output_file}")
