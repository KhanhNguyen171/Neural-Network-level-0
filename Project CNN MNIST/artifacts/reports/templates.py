from datetime import datetime


def create_evaluation_report(
    metrics: dict,
    model_name: str,
):
    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }