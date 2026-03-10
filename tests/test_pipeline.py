"""Integration tests for the detection pipeline."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import DetectorResult, LayerResult
from detection.pipeline import Pipeline


class DummyLayer(BaseLayer):
    name = "dummy"
    order = 0.0

    def __init__(self, score=0.5, **kwargs):
        super().__init__(kwargs.get("config", {}), kwargs.get("device", "cpu"))
        self._score = score

    def setup(self):
        pass

    def analyze(self, image, context):
        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=self._score,
            detector_results=[
                DetectorResult(detector_name="dummy_det", score=self._score, confidence=0.8)
            ],
        )


def test_pipeline_runs_with_dummy_layers():
    config = {"pipeline": {"max_parallel_detectors": 2}}
    pipeline = Pipeline(config)

    layer1 = DummyLayer(score=0.3, config={}, device="cpu")
    layer1.name = "layer_a"
    layer1.order = 0.0

    layer2 = DummyLayer(score=0.7, config={}, device="cpu")
    layer2.name = "layer_b"
    layer2.order = 1.0

    pipeline.register_layer(layer1)
    pipeline.register_layer(layer2)
    pipeline.setup()

    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = pipeline.run(image)

    assert result.verdict in ("authentic", "uncertain", "ai_generated")
    assert 0 <= result.overall_score <= 1
    assert len(result.layer_results) == 2
    assert result.processing_time_ms > 0


def test_pipeline_empty():
    pipeline = Pipeline({})
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = pipeline.run(image)
    assert result.verdict == "authentic"


def test_pipeline_layer_ordering():
    pipeline = Pipeline({})

    l1 = DummyLayer(score=0.1, config={}, device="cpu")
    l1.name = "first"
    l1.order = 2.0

    l2 = DummyLayer(score=0.2, config={}, device="cpu")
    l2.name = "second"
    l2.order = 0.5

    pipeline.register_layer(l1)
    pipeline.register_layer(l2)

    # Should be sorted by order
    assert pipeline.layers[0].name == "second"
    assert pipeline.layers[1].name == "first"
