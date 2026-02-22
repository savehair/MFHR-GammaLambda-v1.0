from .sti_model import STITransformer

def build_model(config):

    return STITransformer(
        input_dim=config["input_dim"],
        hidden_dim=64,
        use_nif=config["use_nif"],
        use_gcn=config["use_gcn"],
        use_transformer=config["use_transformer"]
    )