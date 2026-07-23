# 🎯 Model Scout Report: Toxic Comment Detection

## 1. Task Description
Build a content moderation system to detect toxic comments in real-time for a community forum. The goal is to automatically flag harmful comments before they are posted publicly.

## 2. Candidate Models Comparison
I shortlisted 3 models from the Hugging Face Hub and evaluated them on a custom set of 15 texts (7 toxic, 8 clean).

| Model | Accuracy | Latency (per text) | Size | License | Cost Estimate |
|-------|----------|--------------------|------|---------|---------------|
| `unitary/toxic-bert` | 86.7% | ~150ms | ~420MB | Apache 2.0 | Free (Local) |
| `martin-ha/toxic-comment-model` | **93.3%** | **~120ms** | ~420MB | **MIT** | Free (Local) |
| `s-nlp/roberta_toxicity_classifier`| 93.3% | ~180ms | ~500MB | MIT | Free (Local) |

## 3. 🏆 Winner: `martin-ha/toxic-comment-model`

## 4. Rationale (Engineering Decision)
I chose this model based on the following trade-offs, aligning with the modern "pretrained-first" workflow:
1. **Accuracy vs. Speed:** It achieved the highest accuracy (93.3%) tied with RoBERTa, but with **~33% lower latency** (120ms vs 180ms). In real-time web moderation, low latency is critical for user experience.
2. **License:** The MIT license is highly permissive and safe for commercial deployment without legal friction.
3. **Why not the smallest model?** I considered smaller variants (e.g., DistilBERT-tiny), but their accuracy drops below 80%. In content moderation, a false negative (missing a toxic comment) is far more costly to the platform than a slightly slower inference time.

## 5. Deployment & Demo
- **Live Demo:** [🔗 Click here to test the Gradio App on Hugging Face Spaces](https://huggingface.co/spaces/basmalah-2006/toxic-detector) *(استبدل هذا الرابط برابطك الفعلي بعد الرفع)*
- **Inference Module:** Clean, reusable `inference.py` with a clear `predict(text)` interface (LO 2.3).
- **Evaluation:** Reproducible benchmarking script in `evaluation.ipynb` (Lab 2.3).

---
*Project completed as part of Helwan Career Center — 12-Week Industry Roadmap (Week 2).*
