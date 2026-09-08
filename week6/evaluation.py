import csv
import time
import re
from pathlib import Path

try:
    from rag_chain import rag_chain as baseline_chain, retriever as baseline_retriever
    from rag_chain_reranker import rag_chain as reranker_chain, reranking_retriever
    from rag_chain_hybrid import rag_chain as hybrid_chain, final_retriever as hybrid_retriever
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all rag_chain_*.py files are in the same directory.")
    exit(1)

EVAL_DATASET = [
    {
        "question": "ما هي أهم عوامل الخطر التي تؤثر على الصحة النفسية في بيئة العمل؟",
        "expected_refusal": False
    },
    {
        "question": "كيف يمكن للمنظمة تعزيز الرفاه النفسي للموظفين؟",
        "expected_refusal": False
    },
    {
        "question": "ما هي علامات الإرهاق الوظيفي التي يجب أن ينتبه لها المشرفون؟",
        "expected_refusal": False
    },
    {
        "question": "How can a manager support employees mentally?",
        "expected_refusal": False
    },
    
    {
        "question": "ما هو اضطراب القلق العام GAD؟",
        "expected_refusal": True
    },
    {
        "question": "What is PTSD and how is it diagnosed?",
        "expected_refusal": True
    },
    {
        "question": "ما هي معايير DSM-5 لتشخيص الاكتئاب الحاد؟",
        "expected_refusal": True
    },
]

def check_citations(answer: str) -> bool:
    """Checks if the answer contains page citations like (Page: X) or (المصدر: الصفحة X)."""
    pattern = r"(Page:\s*\d+|الصفحة\s*\d+|المصدر)"
    return bool(re.search(pattern, answer, re.IGNORECASE))

def check_refusal(answer: str) -> bool:
    """Checks if the answer is a refusal message."""
    refusal_phrases = [
        "I do not have enough information",
        "لا أملك معلومات كافية"
    ]
    return any(phrase.lower() in answer.lower() for phrase in refusal_phrases)

def run_evaluation(chain, chain_name: str, results_list: list):
    """Runs the evaluation for a specific chain and appends results to the list."""
    print(f"\n{'='*60}")
    print(f"🔬 Evaluating: {chain_name}")
    print(f"{'='*60}")
    
    citation_hits = 0
    refusal_hits = 0
    total_latency = 0
    n_valid = 0
    
    for item in EVAL_DATASET:
        q = item["question"]
        expected_refusal = item["expected_refusal"]
        
        print(f"  ⏳ Processing: {q[:50]}...")
        start_time = time.time()
        
        try:
            answer = chain.invoke(q)
            latency = round(time.time() - start_time, 2)
            total_latency += latency
            
            has_citation = check_citations(answer)
            is_refusal = check_refusal(answer)
            
            citation_correct = (has_citation and not expected_refusal) or (not has_citation and expected_refusal)
            refusal_correct = (is_refusal == expected_refusal)
            
            n_valid += 1
            if citation_correct:
                citation_hits += 1
            if refusal_correct:
                refusal_hits += 1
            
            results_list.append({
                "chain_version": chain_name,
                "question": q,
                "expected_refusal": "Yes" if expected_refusal else "No",
                "generated_answer": answer.replace("\n", " ")[:150] + "...",
                "has_citation": "Yes" if has_citation else "No",
                "correct_citation": "✅" if citation_correct else "❌",
                "correct_refusal": "✅" if refusal_correct else "❌",
                "latency_seconds": latency
            })
            
            print(f"    Latency: {latency}s | Citation: {'✅' if citation_correct else '❌'} | Refusal: {'✅' if refusal_correct else '❌'}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results_list.append({
                "chain_version": chain_name,
                "question": q,
                "expected_refusal": "Yes" if expected_refusal else "No",
                "generated_answer": f"ERROR: {e}",
                "has_citation": "No",
                "correct_citation": "❌",
                "correct_refusal": "❌",
                "latency_seconds": 0
            })

    if n_valid > 0:
        citation_accuracy = (citation_hits / n_valid) * 100
        refusal_accuracy = (refusal_hits / n_valid) * 100
        avg_latency = total_latency / n_valid
        
        print(f"\n  📊 Summary for {chain_name}:")
        print(f"     Citation Accuracy: {citation_accuracy:.1f}%")
        print(f"     Refusal Accuracy: {refusal_accuracy:.1f}%")
        print(f"     Avg Latency: {avg_latency:.2f}s")
    else:
        print(f"\n  ❌ All queries failed for {chain_name}")

if __name__ == "__main__":
    all_results = []
    
    run_evaluation(baseline_chain, "Baseline", all_results)
    run_evaluation(reranker_chain, "Re-ranker", all_results)
    run_evaluation(hybrid_chain, "Hybrid", all_results)
    
    # 4. Save Results to CSV
    output_file = Path(__file__).parent / "evaluation_results.csv"
    
    if all_results:
        keys = all_results[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
        print(f"\n{'='*60}")
        print(f"🎉 Evaluation Complete!")
        print(f"📊 Results saved to: {output_file}")
        print(f"{'='*60}")