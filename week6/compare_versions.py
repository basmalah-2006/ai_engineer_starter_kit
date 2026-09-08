from rag_chain import rag_chain as baseline_chain
from rag_chain_reranker import rag_chain as reranker_chain
from rag_chain_hybrid import rag_chain as hybrid_chain

test_questions = [
    "ما هو اضطراب القلق العام GAD؟",
    "What is PTSD and how is it diagnosed?",
    "كيف يمكن للمدير أن يدعم موظفيه نفسياً؟",
    "ما هي معايير DSM-5 لتشخيص الاكتئاب الحاد؟",
    "ما الفرق بين الإرهاق الوظيفي Burnout والاكتئاب Depression؟"
]

print("="*80)
print("📊 COMPARISON: Baseline vs Re-ranker vs Hybrid")
print("="*80)

for q in test_questions:
    print(f"\n❓ Question: {q}")
    print("-" * 80)
    
    print("📌 [1] Baseline Answer:")
    print(baseline_chain.invoke(q))
    
    print("\n📌 [2] Re-ranker Answer:")
    print(reranker_chain.invoke(q))
    
    print("\n📌 [3] Hybrid Answer:")
    print(hybrid_chain.invoke(q))
    
    print("\n" + "="*80)