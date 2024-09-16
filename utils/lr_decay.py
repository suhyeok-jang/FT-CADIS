# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


def param_groups_lrd(
    model,
    weight_decay=0.05,
    no_weight_decay_list=[
        "classifier.vit.embeddings.cls_token",
        "classifier.vit.embeddings.position_embeddings",
    ],
    layer_decay=0.75,
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    # ViT-B
    num_layers = len(model.classifier.vit.encoder.layer) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # following timm: set wd as 0 for bias, norm layers, cls_token and pos_embedding
        if p.ndim == 1 or n.endswith(".bias") or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def param_groups_lora_lrd(model, weight_decay=0.05, no_weight_decay_list=None, layer_decay=0.75):

    param_group_names = {}
    param_groups = {}

    # ViT-B
    num_layers = len(model.classifier.lora_vit.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # following timm: set wd as 0 for bias, norm layers, cls_token and pos_embedding
        if p.ndim == 1 or n.endswith(".bias") or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:  # Lora rank matrix weight, FC layer weight
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit_lora(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit_lora(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["classifier.lora_vit.cls_token", "classifier.lora_vit.pos_embed"]:
        return 0
    elif name.startswith("classifier.lora_vit.patch_embed"):
        return 0
    elif name.startswith("classifier.lora_vit.blocks"):
        # Number of layer
        return int(name.split(".")[3]) + 1
    else:  # FC layer
        return num_layers


def get_layer_id_for_vit(name, num_layers):

    if name in [
        "classifier.vit.embeddings.cls_token",
        "classifier.vit.embeddings.position_embeddings",
    ]:
        return 0
    elif name.startswith("classifier.vit.embeddings.patch_embeddings"):
        return 0
    elif name.startswith("classifier.vit.encoder.layer"):
        # Number of layer
        return int(name.split(".")[4]) + 1
    else:  # FC layer
        return num_layers
