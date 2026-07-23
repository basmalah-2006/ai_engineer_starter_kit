import gradio as gr
from inference import ToxicCommentDetector

detector = ToxicCommentDetector("martin-ha/toxic-comment-model")

def analyze_comment(text):
    if not text.strip():
        return "⚠️ Please enter a comment first."
    
    result = detector.predict(text)
    emoji = "🚫" if result['is_toxic'] else "✅"
    color = "red" if result['is_toxic'] else "green"
    verdict = "TOXIC COMMENT DETECTED" if result['is_toxic'] else "CLEAN COMMENT"
    
    return f"## {emoji} <span style='color:{color}'>{verdict}</span>\n\n**Confidence:** {result['confidence']:.2%}"

demo = gr.Interface(
    fn=analyze_comment,
    inputs=gr.Textbox(label="Enter a comment to analyze", lines=3, placeholder="Type here..."),
    outputs=gr.Markdown(label="Analysis Result"),
    title="🛡️ Toxic Comment Detector",
    description="Detect toxic or harmful comments using AI. Built for Helwan Career Center Week 2 Project.",
    examples=[["You're an amazing person, thanks!"], ["You are a complete idiot and I hate you."]]
)

demo.launch()
