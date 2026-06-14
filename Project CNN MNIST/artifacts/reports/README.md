# Reports

Thư mục chứa các báo cáo cuối cùng của mô hình.

## Cấu trúc

```text
reports/
├── evaluation.json
├── benchmark.json
├── classification_report.txt
└── summary.txt
```

## evaluation.json

```json
{
    "model": "MNISTCNN",
    "timestamp": "...",
    "metrics": {
        "accuracy": 0.992,
        "macro_precision": 0.991,
        "macro_recall": 0.991,
        "macro_f1": 0.991
    }
}
```

## benchmark.json

```json
{
    "parameters": 123456,
    "model_size_mb": 0.52,
    "latency_ms": 1.24,
    "throughput": 800
}
```

## classification_report.txt

Báo cáo per-class precision,
recall và f1-score.

## summary.txt

Tóm tắt toàn bộ kết quả huấn luyện.