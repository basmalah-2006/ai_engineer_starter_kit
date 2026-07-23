# Model Scout Report — Toxic Comment Detector
 
**Task:** Binary classification of user comments as *toxic* or *non-toxic*, for a hypothetical
comment-moderation use case (e.g. flagging comments for review before they're published).
 
## Candidate models
 
| Model | Architecture | Params (approx.) | Notes |
|---|---|---|---|
| `unitary/toxic-bert` | BERT-base | ~110M | Multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate), trained on Jigsaw. |
| `martin-ha/toxic-comment-model` | DistilBERT | ~66M | Binary toxic/non-toxic, distilled — smaller & faster. |
| `s-nlp/roberta_toxicity_classifier` | RoBERTa-base | ~125M | Binary toxic/neutral, trained on Jigsaw + additional toxicity data. |
 
## Benchmark results
 
*Run `model_scout_comparison.ipynb` locally and paste your numbers here.*
 
| Model | Accuracy (15-sentence test set) | Avg. latency (ms/sentence, CPU) | Cost (Params / RAM) |
|---|---|---|---|
| unitary/toxic-bert | 53.3% | 53.3 | High (~110M params) |
| martin-ha/toxic-comment-model | 73.3% | 29.0 | Lowest (~66M params) |
| s-nlp/roberta_toxicity_classifier | **100%** | 40.2 | Medium (~125M params) |
 
## Chosen model: `s-nlp/roberta_toxicity_classifier`
 
**Justification:**
 
`s-nlp/roberta_toxicity_classifier` was the clear winner, correctly classifying all 15
sentences in the test set (100% accuracy) versus 73.3% for `martin-ha/toxic-comment-model`
and only 53.3% for `unitary/toxic-bert`. Its latency (40.2ms/sentence on CPU) is roughly
in between the other two — slower than the distilled `martin-ha` model (29.0ms) but notably
faster than `unitary/toxic-bert` (53.3ms) — so it does not sacrifice speed to get the accuracy
gain, making it the best overall trade-off for a production moderation pipeline. `unitary/toxic-bert`'s
weaker score is likely explained by its multi-label design: it was trained to output separate
scores per category (toxic, severe_toxic, obscene, threat, insult, identity_hate) rather than
a single binary toxic/non-toxic decision, so collapsing its output into a binary label with a
simple "any label above threshold" rule likely loses some of the signal it was actually trained
to provide. `martin-ha/toxic-comment-model` was fastest, since DistilBERT is roughly half the
parameter count of the RoBERTa/BERT-base models, which is a meaningful advantage at real
production scale, but its accuracy gap on this test set was too large to prefer it over the
RoBERTa model.
 
## Reflection
 
- What would you do differently with more time/data (e.g. a larger, more diverse test set;
  testing on real Jigsaw validation data instead of 15 hand-written sentences)?
- What's a real limitation of the winning model you'd flag to a hiring manager (bias, false
  positives on sarcasm, English-only, etc.)?
 
