import torchmetrics


def get_metric(metric_name):
    if hasattr(torchmetrics, metric_name):
        return getattr(torchmetrics, metric_name)
    else:
        raise AttributeError(f"{metric_name=} not found in module {torchmetrics.__name__}")
