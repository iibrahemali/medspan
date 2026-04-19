import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import gradio as gr
import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_DIR = "./saved_model"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.safetensors")

if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(
        "Missing saved_model/model.safetensors. Download the fine-tuned weights "
        "into ./saved_model, or load the model from the Hugging Face Hub instead."
    )

if torch.cuda.is_available():
    PIPELINE_DEVICE = 0
elif torch.backends.mps.is_available():
    PIPELINE_DEVICE = "mps"
else:
    PIPELINE_DEVICE = -1

print(f"Loading model from {MODEL_DIR} …")
qa_pipeline = pipeline(
    "question-answering",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=PIPELINE_DEVICE,
    handle_impossible_answer=False,
)
print("Model loaded.")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def answer_question(question: str, context: str) -> str:
    """Run extractive QA and return the predicted answer span."""
    question = question.strip()
    context = context.strip()

    if not question or not context:
        return "Please provide both a question and a PubMed abstract."

    result = qa_pipeline(question=question, context=context)
    answer = result.get("answer", "No answer found.")
    score = result.get("score", 0.0)
    return f"{answer}  (confidence: {score:.2%})"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
EXAMPLE_Q = "What is the mechanism of action of metformin in type 2 diabetes?"
EXAMPLE_CTX = (
    "Metformin is a biguanide drug used as a first-line treatment for type 2 "
    "diabetes mellitus. Its primary mechanism of action involves the inhibition "
    "of hepatic glucose production (gluconeogenesis) via activation of AMP-activated "
    "protein kinase (AMPK). Additionally, metformin improves peripheral insulin "
    "sensitivity and reduces intestinal glucose absorption. Unlike sulfonylureas, "
    "metformin does not stimulate insulin secretion and therefore carries a low "
    "risk of hypoglycaemia."
)

with gr.Blocks(title="MedSpan — Biomedical QA") as demo:
    gr.Markdown(
        """
        # MedSpan — Biomedical Question Answering
        Fine-tuned **BioBERT** (dmis-lab/biobert-base-cased-v1.2) on BioASQ factoid questions.
        Paste a PubMed abstract and ask a factoid question to extract the answer span.
        """
    )

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Medical Question",
                placeholder="e.g. What gene is mutated in cystic fibrosis?",
                lines=2,
            )
            context_input = gr.Textbox(
                label="PubMed Abstract",
                placeholder="Paste the relevant PubMed abstract here …",
                lines=8,
            )
            submit_btn = gr.Button("Extract Answer", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Extracted Answer Span", lines=3)

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, context_input],
        outputs=answer_output,
    )

    gr.Examples(
        examples=[[EXAMPLE_Q, EXAMPLE_CTX]],
        inputs=[question_input, context_input],
        outputs=answer_output,
        fn=answer_question,
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---
        **Model:** dmis-lab/biobert-base-cased-v1.2 fine-tuned on BioASQ Task B (factoid)
        **Task:** Extractive span extraction from PubMed abstracts
        **Source:** [GitHub](https://github.com/ibrahemali/medspan)
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
