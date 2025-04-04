from loguru import logger

import torch

import ml.optim.custom.optim as custom_optim


def init_optims_from_config(config, model):
    custom_keys_weight_decay = [
        (key, 0.0) for key in ["class_token", "position_embedding", "relative_position_bias_table"]
    ]
    if hasattr(model, "custom_keys_weight_decay_filter"):
        custom_keys_weight_decay.extend(
            [(key, 0.0) for key in model.custom_keys_weight_decay_filter]
        )

    group_names, params = set_weight_decay(
        model,
        config.opt.params.weight_decay,
        0.0,
        custom_keys_weight_decay=custom_keys_weight_decay,
    )
    logger.info("Turning Off Norm Weight Decay")
    for group_name, param_groups in zip(group_names, params):
        logger.info(
            f"{group_name}: "
            f"{len(param_groups['params'])} parameters have weight decay: {param_groups['weight_decay']}"
        )

    if hasattr(torch.optim, config.opt.type):
        opt = getattr(torch.optim, config.opt.type)(params, **config.opt.params)
    elif hasattr(custom_optim, config.opt.type) and config.opt.type != "SAM":
        opt = getattr(custom_optim, config.opt.type)(params, **config.opt.params)
    elif hasattr(custom_optim, config.opt.type) and config.opt.type == "SAM":
        base_optim_cls = getattr(torch.optim, config.opt.type)

        opt = getattr(custom_optim, config.opt.type)(
            params, base_optim=base_optim_cls, **config.opt.params
        )
    else:
        raise NotImplementedError(f"Unknown optimizer: {config.opt.type}")

    return [opt], group_names


def set_weight_decay(
    model,
    weight_decay,
    norm_weight_decay=None,
    norm_classes=None,
    custom_keys_weight_decay=None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
            torch.nn.modules.batchnorm._NormBase,
            torch.nn.modules.batchnorm._LazyNormBase,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
        "bias": [],
    }

    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
        "bias": 0.0,
    }

    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False

            if name.endswith("bias"):
                params["bias"].append(p)
                continue

            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break

            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    # logger.info(f"{params}, {params_weight_decay}, {custom_keys}, {norm_classes}, {weight_decay}, {norm_weight_decay}")

    param_group_names = []
    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_group_names.append(key)
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_group_names, param_groups


if __name__ == "__main__":
    import torchinfo
    from ml_collections import ConfigDict

    bleh_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 6 * 6, 256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10, bias=False),
    )
    # use torchinfo to get the model summary
    model_info = torchinfo.summary(bleh_model, verbose=0)
    print(model_info)

    _config = ConfigDict(
        {
            "opt": {
                "type": "Adam",
                "params": {"lr": 0.001, "weight_decay": 0.01},
            }
        }
    )

    init_optims_from_config(_config, bleh_model)
