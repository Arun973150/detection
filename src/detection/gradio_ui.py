"""Gradio web interface for interactive image authenticity detection."""

import json
import logging

import gradio as gr
import numpy as np

from detection.config import load_config
from detection.app import build_pipeline
from detection.utils.heatmap import create_heatmap_overlay
from detection.utils.image_io import load_image

logger = logging.getLogger(__name__)

pipeline = None


def init_pipeline():
    global pipeline
    if pipeline is None:
        config = load_config()
        pipeline = build_pipeline(config)


def analyze_image(image: np.ndarray):
    """Analyze an uploaded image and return results."""
    if image is None:
        return None, "No image provided", "{}"

    init_pipeline()

    result = pipeline.run(image)

    # Create heatmap overlay
    overlay = None
    if result.merged_heatmap is not None:
        overlay = create_heatmap_overlay(image, result.merged_heatmap, alpha=0.4)

    # Format verdict text
    verdict_text = _format_verdict(result)

    # Full breakdown JSON
    breakdown = json.dumps(result.to_dict(), indent=2, default=str)

    return overlay, verdict_text, breakdown


def _format_verdict(result) -> str:
    """Format the pipeline result into a readable verdict."""
    lines = []

    # Main verdict
    verdict_emoji = {
        "authentic": "AUTHENTIC",
        "ai_generated": "AI-GENERATED",
        "manipulated": "MANIPULATED",
        "deepfake": "DEEPFAKE",
        "uncertain": "UNCERTAIN",
    }
    label = verdict_emoji.get(result.verdict, result.verdict.upper())
    lines.append(f"VERDICT: {label}")
    lines.append(f"Confidence: {result.confidence:.1%}")
    lines.append(f"Overall Score: {result.overall_score:.4f}")
    lines.append(f"Processing Time: {result.processing_time_ms:.0f}ms")

    # Flags
    if result.flags:
        lines.append(f"\nFlags: {', '.join(result.flags)}")

    # Layer breakdown
    lines.append("\n--- Layer Breakdown ---")
    for lr in result.layer_results:
        lines.append(f"\n{lr.layer_name} (score: {lr.aggregated_score:.4f})")
        if lr.flags:
            lines.append(f"  Flags: {', '.join(lr.flags)}")
        for dr in lr.detector_results:
            status = f"  {dr.detector_name}: {dr.score:.4f}"
            if dr.error:
                status += f" [ERROR: {dr.error}]"
            lines.append(status)

    return "\n".join(lines)


def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="Image Authenticity Detection") as demo:
        gr.Markdown("# Image Authenticity Detection System")
        gr.Markdown(
            "Upload an image to analyze whether it is real, AI-generated, "
            "manipulated, or a deepfake. The system runs multiple detection "
            "layers and produces a fused verdict with confidence score."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                output_heatmap = gr.Image(label="Suspicion Heatmap", type="numpy")

        with gr.Row():
            verdict_output = gr.Textbox(
                label="Verdict & Layer Breakdown",
                lines=20,
                max_lines=40,
            )

        with gr.Accordion("Full JSON Breakdown", open=False):
            json_output = gr.Code(language="json", label="Raw Results")

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_heatmap, verdict_output, json_output],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
