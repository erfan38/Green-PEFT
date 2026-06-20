# energypeft/integrations/__init__.py
"""
Optional integrations for third-party frameworks.

These modules have optional dependencies:
  - huggingface_peft: requires `peft` package
  - llamafactory: requires `llamafactory` package
  - transformers: requires `transformers` package (usually installed)
"""

__all__ = []

# HuggingFace PEFT (optional)
try:
    from .huggingface_peft import HuggingFacePEFTTrainer, get_default_lora_config
    __all__.extend(["HuggingFacePEFTTrainer", "get_default_lora_config"])
except ImportError:
    HuggingFacePEFTTrainer = None
    get_default_lora_config = None

# LlamaFactory (optional)
try:
    from .llamafactory import LlamaFactoryEnergyWrapper, LlamaFactoryNotFoundError
    __all__.extend(["LlamaFactoryEnergyWrapper", "LlamaFactoryNotFoundError"])
except ImportError:
    LlamaFactoryEnergyWrapper = None
    LlamaFactoryNotFoundError = None

# Transformers (optional but usually installed)
try:
    from .transformers import TransformersTrainer
    __all__.append("TransformersTrainer")
except ImportError:
    TransformersTrainer = None
