import csv
import json
from pathlib import Path
from statistics import mean

from semantic_sentence_postprocessor import semantic_sentence_postprocess


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "sentence_degree_test_cases.csv"

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUT_DIR / "sentence_degree_eval_results.csv"
METRICS_JSON = OUT_DIR / "sentence_degree_eval_metrics.json"
REPORT_TXT = OUT_DIR / "sentence_degree_eval_report.txt"
FAIL_CASES_CSV = OUT_DIR / "sentence_degree_fail_cases.csv"


def normalize_text(text: str) -> str:
    return str(text).strip().replace("  ", " ")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"테스트 케이스 파일이 없습니다: {INPUT_CSV}")

    rows = []

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "id",
            "raw_text",
            "degree",
            "degree_ko",
            "expected_final_text",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 컬럼이 부족합니다: {sorted(missing)}")

        for row in reader:
            sentence_result = {
                "text": row["raw_text"],
                "confidence": 1.0,
            }

            degree_result = {
                "degree": row["degree"],
                "degree_ko": row["degree_ko"],
                "confidence": 1.0,
            }

            semantic_result = semantic_sentence_postprocess(
                sentence_result=sentence_result,
                degree_result=degree_result,
                use_llm=True,
            )

            expected = normalize_text(row["expected_final_text"])
            actual = normalize_text(semantic_result.get("final_text", ""))

            is_pass = actual == expected

            rows.append({
                "id": row["id"],
                "raw_text": row["raw_text"],
                "degree": row["degree"],
                "degree_ko": row["degree_ko"],
                "expected_final_text": expected,
                "actual_final_text": actual,
                "pass": is_pass,
                "apply_degree": semantic_result.get("apply_degree", ""),
                "target_expression": semantic_result.get("target_expression", ""),
                "modifier": semantic_result.get("modifier", ""),
                "source": semantic_result.get("source", ""),
                "reason": semantic_result.get("reason", ""),
                "latency_ms": semantic_result.get("latency_ms", ""),
            })

    if not rows:
        raise ValueError("테스트 케이스가 비어 있습니다.")

    total = len(rows)
    pass_count = sum(1 for r in rows if r["pass"])
    fail_count = total - pass_count

    source_counts = {}
    for r in rows:
        source = r["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    ollama_count = source_counts.get("ollama", 0)
    fallback_count = source_counts.get("fallback", 0)
    rule_count = source_counts.get("rule", 0)
    rule_guard_count = source_counts.get("rule_guard", 0)

    latency_values = []
    for r in rows:
        value = r.get("latency_ms", "")
        if value != "":
            try:
                latency_values.append(float(value))
            except ValueError:
                pass

    postprocess_accuracy = pass_count / total
    ollama_final_accept_rate = ollama_count / total
    fallback_rate = fallback_count / total
    rule_rate = rule_count / total
    rule_guard_rate = rule_guard_count / total
    avg_latency_ms = mean(latency_values) if latency_values else 0.0

    fieldnames = list(rows[0].keys())

    with open(RESULT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fail_rows = [r for r in rows if not r["pass"]]
    with open(FAIL_CASES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fail_rows)

    metrics = {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "postprocess_accuracy": round(postprocess_accuracy, 4),
        "ollama_final_accept_rate": round(ollama_final_accept_rate, 4),
        "rule_guard_rate": round(rule_guard_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "rule_rate": round(rule_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "source_counts": source_counts,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = f"""
문장-degree 후처리 검증 리포트
================================

총 테스트 케이스: {total}
PASS: {pass_count}
FAIL: {fail_count}

후처리 정확도: {postprocess_accuracy:.4f}
Ollama 최종 채택률: {ollama_final_accept_rate:.4f}
Rule Guard 안전 보정률: {rule_guard_rate:.4f}
Fallback 사용률: {fallback_rate:.4f}
Rule 처리율: {rule_rate:.4f}
평균 지연시간(ms): {avg_latency_ms:.2f}

Source 분포:
{json.dumps(source_counts, ensure_ascii=False, indent=2)}

생성 파일:
- {RESULT_CSV}
- {METRICS_JSON}
- {REPORT_TXT}
- {FAIL_CASES_CSV}
""".strip()

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()