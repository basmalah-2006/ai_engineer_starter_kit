from transformers import pipeline
import torch

class ToxicCommentDetector:
    def __init__(self, model_name: str = "martin-ha/toxic-comment-model"):
        # استخدام GPU لو متاح، وإلا CPU
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            truncation=True
        )
        self.model_name = model_name
    
    def predict(self, text: str) -> dict:
        """
        Predict if a comment is toxic or not.
        Args:
            text: The comment text to analyze
        Returns:
            dict with keys: 'is_toxic' (bool), 'confidence' (float), 'label' (str)
        """
        result = self.classifier(text)[0]
        
        # توحيد مسميات المخرجات لأن كل نموذج بيستخدم مسمى مختلف
        label = result['label'].lower()
        is_toxic = label in ['toxic', 'label_1', '1']
        
        return {
            "is_toxic": is_toxic,
            "confidence": result['score'],
            "label": "TOXIC" if is_toxic else "CLEAN"
        }

# كود للتجربة السريعة
if __name__ == "__main__":
    detector = ToxicCommentDetector()
    print(detector.predict("You are an amazing person!"))
    print(detector.predict("You are a complete idiot"))
