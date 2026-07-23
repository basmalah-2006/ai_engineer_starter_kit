from transformers import pipeline
 
# Winner model chosen after benchmarking in model_scout_comparison.ipynb.
# s-nlp/roberta_toxicity_classifier scored 100% accuracy on the 15-sentence test set
# (vs. 73.3% for martin-ha/toxic-comment-model and 53.3% for unitary/toxic-bert),
# with a mid-range latency of ~40ms/sentence on CPU.
MODEL_NAME = "s-nlp/roberta_toxicity_classifier"
 
_classifier = None
 
 
def _get_classifier():
    """Lazily load the pipeline once and reuse it across calls."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=MODEL_NAME)
    return _classifier
 
 
def predict(text: str) -> dict:
    """Classify a single piece of text as toxic or non-toxic.
 
    Args:
        text: the comment/sentence to classify.
 
    Returns:
        A dict like {"label": "toxic", "score": 0.987} where label is
        normalized to lowercase "toxic" / "non-toxic".
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")
 
    clf = _get_classifier()
    result = clf(text)[0]
 
    label = result["label"].lower()
    if "non" in label or "neutral" in label or label in ("label_0", "clean"):
        label = "non-toxic"
    else:
        label = "toxic"
 
    return {"label": label, "score": round(float(result["score"]), 4)}
 
 
if __name__ == "__main__":
    # Quick manual smoke test: python inference.py
    samples = [
        "You are an idiot and everyone hates you.",
        "Thanks so much for your help today!",
    ]
    for s in samples:
        print(s, "->", predict(s))