# Benchmark & Model Justification

## 📊 Short Benchmark Note
- **Model Used:** Llama 3.1 8B (4-bit quantized)
- **Hardware:** Laptop with 12GB RAM, Intel Core i5, NVIDIA GeForce RTX 2050 (4GB Dedicated VRAM) and Shared System RAM
- **Performance:** 
  - Average Speed: ~9 tokens/sec.
- **Quality:** Excellent for empathetic tasks, mood analysis, and creative summarization. Handles both English and Arabic inputs gracefully.

## 🧠 Model-Choice Justification
I chose **Llama 3.1 8B** for this offline journaling tool because:
1. **Empathy & Creativity:** It has strong instruction-tuning for creative and empathetic writing, which is crucial for a journaling assistant.
2. **Hardware Efficiency:** The 4-bit quantized version fits perfectly within 8GB of RAM/VRAM, ensuring smooth offline inference on standard laptops without freezing the OS.
3. **Multilingual Support:** It performs exceptionally well in both English and Arabic, allowing users to journal in their preferred language.
4. **Open License:** It is fully open-source, aligning perfectly with the "Privacy-First" and offline nature of this project.