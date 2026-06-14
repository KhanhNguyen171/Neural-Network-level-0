# Logs

Thư mục lưu toàn bộ log của hệ thống.

## Cấu trúc

```text
logs/
├── training.log
├── experiment.log
├── evaluation.log
└── deployment.log
```

## training.log

Ví dụ:

```text
2026-06-14 10:00:00 | INFO | Epoch=1 train_loss=0.42 val_loss=0.31 accuracy=0.91
2026-06-14 10:00:15 | INFO | Epoch=2 train_loss=0.21 val_loss=0.14 accuracy=0.96
```

## experiment.log

Ví dụ:

```text
START_EXPERIMENT: baseline
METRIC accuracy=0.991
METRIC macro_f1=0.990
FINISH_EXPERIMENT: baseline
```

## Mục đích

- Theo dõi quá trình huấn luyện
- Debug lỗi
- Audit experiment
- Deployment monitoring