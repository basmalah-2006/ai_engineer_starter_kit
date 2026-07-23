import gradio as gr
from inference import predict, MODEL_NAME


def classify_comment(text: str):
    if not text or not text.strip():
        return "Please enter some text."
    result = predict(text)
    label = result["label"]
    score = result["score"]
    emoji = "🚫" if label == "toxic" else "✅"
    return f"{emoji} {label.upper()}  (confidence: {score:.2%})"


demo = gr.Interface(
    fn=classify_comment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Type a comment to check...",
        label="Comment",
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Toxic Comment Detector",
    description=(
        f"Classifies a comment as toxic or non-toxic. "
        f"Model: {MODEL_NAME} — chosen after comparing 3 candidate models "
        f"(see model_scout_report.md)."
    ),
    examples=[
        ["Thanks for the quick reply, really appreciate it!"],
        ["You are an idiot and should just quit."],
        ["Let's schedule the meeting for Thursday."],
    ],
)

if __name__ == "__main__":
    demo.launch()
