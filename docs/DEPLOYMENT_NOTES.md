# Deployment Notes

Add a Gradio app that accepts an image and returns:

- generated caption
- decoding settings
- optional top-k alternate captions

Serving notes:

- cache CLIP embeddings when possible
- limit upload size
- run on CPU for demos if latency is acceptable
- never claim benchmark quality without logged evaluation artifacts
