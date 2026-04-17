from pathlib import Path
from statistics import mean
from datetime import datetime
import csv
import json

from graph.workflow import workflow


def _pick_value(container, key: str):
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def main() -> None:
    samples_dir = Path("data/samples")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_json_path = reports_dir / f"batch_evaluation_summary_{run_timestamp}.json"
    summary_csv_path = reports_dir / f"batch_evaluation_summary_{run_timestamp}.csv"

    files = sorted(
        [
            path
            for path in samples_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".png", ".jpg", ".jpeg"}
        ]
    )

    results = []

    for index, path in enumerate(files, start=1):
        thread_id = f"batch_samples_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{index}"
        try:
            state = workflow.process_document(str(path), thread_id=thread_id)
            success = bool(state.get("success", False))
            metrics = state.get("metrics") or {}
            extraction_accuracy = _pick_value(metrics, "extraction_accuracy")
            if extraction_accuracy is None:
                extraction_accuracy = state.get("current_accuracy")

            pii_recall = _pick_value(metrics, "pii_recall")
            if pii_recall is None:
                redaction_result = state.get("redaction_result")
                pii_recall = _pick_value(redaction_result, "recall")

            results.append(
                {
                    "file": path.name,
                    "success": success,
                    "extraction_accuracy": extraction_accuracy,
                    "pii_recall": pii_recall,
                    "errors": state.get("errors", []),
                }
            )
            print(
                f"[{index}/{len(files)}] {path.name} -> "
                f"success={success}, extraction_accuracy={extraction_accuracy}, pii_recall={pii_recall}"
            )
        except Exception as exc:
            results.append(
                {
                    "file": path.name,
                    "success": False,
                    "extraction_accuracy": None,
                    "pii_recall": None,
                    "errors": [str(exc)],
                }
            )
            print(f"[{index}/{len(files)}] {path.name} -> FAILED: {exc}")

    successes = [result for result in results if result["success"]]
    workflow_success_rate = (len(successes) / len(results)) if results else 0.0

    extraction_values = [
        result["extraction_accuracy"]
        for result in successes
        if isinstance(result["extraction_accuracy"], (int, float))
    ]
    pii_recall_values = [
        result["pii_recall"]
        for result in successes
        if isinstance(result["pii_recall"], (int, float))
    ]

    avg_extraction = mean(extraction_values) if extraction_values else 0.0
    avg_pii_recall = mean(pii_recall_values) if pii_recall_values else 0.0

    print("\n=== BATCH SUMMARY ===")
    print(f"total_files={len(results)}")
    print(f"success_count={len(successes)}")
    print(f"workflow_success_rate={workflow_success_rate:.4f}")
    print(f"avg_extraction_accuracy={avg_extraction:.4f}")
    print(f"avg_pii_recall={avg_pii_recall:.4f}")

    summary_payload = {
        "run_timestamp": run_timestamp,
        "thresholds": {
            "extraction_accuracy": 0.90,
            "pii_recall": 0.95,
            "workflow_success_rate": 0.90,
        },
        "summary": {
            "total_files": len(results),
            "success_count": len(successes),
            "workflow_success_rate": workflow_success_rate,
            "avg_extraction_accuracy": avg_extraction,
            "avg_pii_recall": avg_pii_recall,
        },
        "results": results,
    }

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "success", "extraction_accuracy", "pii_recall", "errors"],
        )
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["errors"] = "; ".join(item.get("errors", []))
            writer.writerow(row)

    print(f"summary_json={summary_json_path}")
    print(f"summary_csv={summary_csv_path}")

    failed = [result for result in results if not result["success"]]
    if failed:
        print("\n=== FAILURES ===")
        for item in failed:
            first_error = item["errors"][0] if item["errors"] else "Unknown error"
            print(f"- {item['file']}: {first_error}")


if __name__ == "__main__":
    main()
