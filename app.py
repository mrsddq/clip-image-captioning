from __future__ import annotations

import gradio as gr


def status() -> str:
    return "CLIP captioning demo scaffold is ready. Train a decoder checkpoint before deployment."


demo = gr.Interface(fn=status, inputs=None, outputs="text", title="CLIP Captioner")


if __name__ == "__main__":
    demo.launch()
