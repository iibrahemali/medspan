## Deploy to Hugging Face Spaces

1. Create a new HF Space with **SDK: Gradio**.
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/ibrahem1ali/medspan
git push space main
```

3. The Space runner installs `requirements.txt` and launches `app.py` automatically.

> Upload `saved_model/` files to the Space repository or load the model directly from the HF Hub if you've pushed it there.
