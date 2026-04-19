# MedSpan — Biomedical Extractive Question Answering

Fine-tuned **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`) on BioASQ Task B factoid questions.
Given a PubMed abstract and a biomedical question, the model extracts the answer span directly from the text.

---

## Project structure

```
medspan/
├── medspan.ipynb    # Training notebook (runs end-to-end on Colab T4 or M1 Mac)
├── app.py           # Gradio inference UI (deployable to HF Spaces)
├── requirements.txt # Pinned dependencies
├── README.md
└── saved_model/     # Created automatically after training
```

---

## How it works

| Component | Detail |
|-----------|--------|
| **Task** | Extractive QA — span extraction from PubMed abstracts |
| **Model** | `dmis-lab/biobert-base-cased-v1.2` + `AutoModelForQuestionAnswering` head |
| **Dataset** | BioASQ Task B, factoid subset via `bigbio/bioasq` on HuggingFace |
| **Tokenization** | Sliding window with stride=128, max_length=384 |
| **Metrics** | Exact Match (EM) & token-level F1 |

---

## Train on Google Colab (T4 GPU)

1. Open `medspan.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to **GPU → T4** via *Runtime → Change runtime type*.
3. Run all cells — the notebook auto-detects CUDA and trains on the full dataset.
4. The fine-tuned model is saved to `saved_model/`.

> **Expected training time:** ~25–40 minutes for 3 epochs on a Colab T4.

---

## Run the UI locally

### Prerequisites

```bash
pip install -r requirements.txt
```

Make sure `saved_model/` contains the fine-tuned model (run the notebook first, or download a pretrained checkpoint).

### Launch

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Deploy to Hugging Face Spaces

1. Create a new HF Space with **SDK: Gradio**.
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/medspan
git push space main
```

3. The Space runner installs `requirements.txt` and launches `app.py` automatically.

> Upload your `saved_model/` files to the Space repository or load the model directly from the HF Hub if you've pushed it there.

---

## M1 Mac (debug mode)

The notebook auto-detects Apple MPS and switches to a 100-example debug subset with 1 training epoch, making it possible to verify the pipeline locally without a GPU.

---

## Citation

```
@article{tsatsaronis2015bioasq,
  title={An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition},
  author={Tsatsaronis, George and others},
  journal={BMC Bioinformatics},
  year={2015}
}

@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and others},
  journal={Bioinformatics},
  year={2020}
}
```
