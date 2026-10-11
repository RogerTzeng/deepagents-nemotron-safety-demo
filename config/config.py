from pathlib import Path


def init(llmrails):
    policy = (Path(__file__).resolve().parent.parent / "policy.txt").read_text().strip()
    if not policy:
        raise ValueError("Company policy must not be empty")
    # Initialization runs before NeMo creates its provider models.
    for model in llmrails.config.models:
        if model.type == "content_safety":
            model.parameters["chat_template_kwargs"]["custom_policy"] = policy
