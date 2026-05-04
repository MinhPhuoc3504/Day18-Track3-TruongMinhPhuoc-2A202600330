"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Thử dùng RAGAS thực (cần OpenAI API key)
    try:
        from ragas import evaluate
        try:
            from ragas.metrics.collections import (faithfulness, answer_relevancy,
                                                    context_precision, context_recall)
        except ImportError:
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        # Bước 1: Tạo Dataset từ các list đầu vào
        # RAGAS yêu cầu đúng 4 keys: question, answer, contexts, ground_truth
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Bước 2: Chạy evaluation với 4 metrics chính
        # - faithfulness: câu trả lời có dựa trên context không? (chống hallucination)
        # - answer_relevancy: câu trả lời có liên quan đến câu hỏi không?
        # - context_precision: context được retrieve có liên quan không?
        # - context_recall: context có chứa đủ thông tin để trả lời không?
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )

        # Bước 3: Chuyển kết quả thành DataFrame và extract scores
        df = result.to_pandas()

        # Bước 4: Tạo per-question EvalResult objects
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            ))

        # Bước 5: Tính aggregate scores (trung bình toàn bộ dataset)
        return {
            "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else 0.0,
            "answer_relevancy": float(result["answer_relevancy"]) if "answer_relevancy" in result else 0.0,
            "context_precision": float(result["context_precision"]) if "context_precision" in result else 0.0,
            "context_recall": float(result["context_recall"]) if "context_recall" in result else 0.0,
            "per_question": per_question,
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        print("  → Falling back to heuristic scoring...")
        return _heuristic_evaluate(questions, answers, contexts, ground_truths)


def _heuristic_evaluate(questions: list[str], answers: list[str],
                         contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Heuristic evaluation khi RAGAS không khả dụng."""
    per_question = []

    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        a_lower = a.lower()
        gt_lower = gt.lower()
        ctx_combined = " ".join(ctx).lower()

        # Faithfulness: câu trả lời có từ nào từ context không?
        answer_words = set(a_lower.split())
        ctx_words = set(ctx_combined.split())
        faithfulness = min(len(answer_words & ctx_words) / max(len(answer_words), 1), 1.0)

        # Answer relevancy: từ trong GT có trong answer không?
        gt_words = set(gt_lower.split())
        answer_relevancy = min(len(gt_words & answer_words) / max(len(gt_words), 1), 1.0)

        # Context precision: từ trong GT có trong context không?
        context_precision = min(len(gt_words & ctx_words) / max(len(gt_words), 1), 1.0)

        # Context recall: answer có trong context không?
        context_recall = min(len(answer_words & ctx_words) / max(len(answer_words), 1), 1.0)

        per_question.append(EvalResult(
            question=q, answer=a, contexts=ctx, ground_truth=gt,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        ))

    def avg(metric):
        vals = [getattr(r, metric) for r in per_question]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    # Bước 1: Tính average score cho mỗi question (trung bình 4 metrics)
    def avg_score(r: EvalResult) -> float:
        return (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4.0

    # Bước 2: Sắp xếp theo avg_score tăng dần → lấy bottom_n failures
    sorted_results = sorted(eval_results, key=avg_score)
    worst = sorted_results[:bottom_n]

    # Bước 3: Phân tích từng failure theo Diagnostic Tree
    failures = []
    for result in worst:
        # Tìm metric tệ nhất
        metric_scores = {
            "faithfulness": result.faithfulness,
            "context_recall": result.context_recall,
            "context_precision": result.context_precision,
            "answer_relevancy": result.answer_relevancy,
        }
        worst_metric = min(metric_scores, key=metric_scores.get)
        worst_score = metric_scores[worst_metric]

        # Map metric tệ nhất → diagnosis và suggested_fix theo Diagnostic Tree
        if worst_metric == "faithfulness" or result.faithfulness < 0.85:
            diagnosis = "LLM hallucinating — trả lời không dựa trên context"
            suggested_fix = "Tighten prompt: yêu cầu LLM chỉ trả lời dựa trên context, lower temperature"
            worst_metric = "faithfulness"
            worst_score = result.faithfulness
        elif worst_metric == "context_recall" or result.context_recall < 0.75:
            diagnosis = "Missing relevant chunks — chunks chứa câu trả lời không được retrieve"
            suggested_fix = "Improve chunking (nhỏ hơn/semantic) hoặc thêm BM25 vào hybrid search"
            worst_metric = "context_recall"
            worst_score = result.context_recall
        elif worst_metric == "context_precision" or result.context_precision < 0.75:
            diagnosis = "Too many irrelevant chunks — context chứa nhiều noise"
            suggested_fix = "Add reranking hoặc metadata filter để loại bỏ chunks không liên quan"
            worst_metric = "context_precision"
            worst_score = result.context_precision
        else:
            diagnosis = "Answer doesn't match question — câu trả lời không trả lời đúng câu hỏi"
            suggested_fix = "Improve prompt template: cụ thể hóa format output và yêu cầu"
            worst_metric = "answer_relevancy"
            worst_score = result.answer_relevancy

        failures.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "avg_score": avg_score(result),
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "all_scores": {
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "context_recall": result.context_recall,
            }
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
