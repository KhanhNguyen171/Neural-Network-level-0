# Metrics

Thư mục lưu kết quả đánh giá của các lần huấn luyện.

## Cấu trúc

```text
metrics/
├── baseline_20260614_100000/
│   ├── metrics.json
│   └── metrics.csv
│
├── cnn_large_20260614_120000/
│   ├── metrics.json
│   └── metrics.csv
```

## metrics.json

```json
{
    "accuracy": 0.992,
    "macro_precision": 0.991,
    "macro_recall": 0.991,
    "macro_f1": 0.991,
    "loss": 0.021
}
```

## Nguồn dữ liệu

Các metric được sinh từ:

- Evaluator
- Trainer.history
- benchmark_model
- classification_report

## Mục đích

- So sánh các experiment
- Theo dõi hiệu năng mô hình
- Phục vụ báo cáo nghiên cứu
- Phục vụ deployment