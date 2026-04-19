## Deploy to Hugging Face Spaces

1. Create a new HF Space with **SDK: Gradio**.
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/ibrahem1ali/medspan
git push space main
```

3. The Space runner installs `requirements.txt` and launches `app.py` automatically.

> Do not commit `saved_model/model.safetensors` to a normal GitHub repository. Put model weights in the HF Space repo with LFS enabled, or load them from the HF Hub.
