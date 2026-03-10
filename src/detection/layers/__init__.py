"""Layer registry — creates all layers from config."""

from detection.base.layer import BaseLayer
from detection.config import get_layer_config


def create_all_layers(config: dict, device: str) -> list[BaseLayer]:
    """Instantiate all enabled layers from config."""
    from detection.layers.layer0_numerical.layer import NumericalPreAnalysisLayer
    from detection.layers.layer0_5_purification.layer import AdversarialPurificationLayer
    from detection.layers.layer1_aigen.layer import AIGeneratedDetectionLayer
    from detection.layers.layer2_localization.layer import ManipulationLocalizationLayer
    from detection.layers.layer2_5_text.layer import TextTypographyLayer
    from detection.layers.layer2_6_semantic.layer import SemanticConsistencyLayer
    from detection.layers.layer3_deepfake.layer import DeepfakeDetectionLayer
    from detection.layers.layer4_watermark.layer import WatermarkDetectionLayer
    from detection.layers.layer5_fusion.layer import EnsembleFusionLayer

    layer_classes = [
        ("layer0_numerical", NumericalPreAnalysisLayer),
        ("layer0_5_purification", AdversarialPurificationLayer),
        ("layer1_aigen", AIGeneratedDetectionLayer),
        ("layer2_localization", ManipulationLocalizationLayer),
        ("layer2_5_text", TextTypographyLayer),
        ("layer2_6_semantic", SemanticConsistencyLayer),
        ("layer3_deepfake", DeepfakeDetectionLayer),
        ("layer4_watermark", WatermarkDetectionLayer),
        ("layer5_fusion", EnsembleFusionLayer),
    ]

    layers = []
    for config_key, cls in layer_classes:
        layer_config = get_layer_config(config, config_key)
        if layer_config.get("enabled", True):
            layers.append(cls(layer_config, device))

    return layers
