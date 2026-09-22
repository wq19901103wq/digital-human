"""完整逐题证据的离线夹具；不伪造只含汇总数字的晋级结果。"""
import json

from src.iteration import datasets, experiment, versions


def finish_with_cases(directory, verdict="merge_to_iteration_baseline", *, wins=2,
                      losses=0, failures=0):
    spec = experiment.spec_of(directory)
    judge = spec["kind"] == "judge_eval"
    if judge:
        path = versions.PRIVATE / "judge_eval" / spec["pack_ref"] / "pack.json"
        rows = json.loads(path.read_text())["rows"]
    else:
        rows = datasets.rows_for(spec)
    if spec.get("smoke"):
        rows = rows[:spec.get("smoke_limit")]
    records = []
    identified_b = identified_c = 0
    for i, row in enumerate(rows):
        b, c = ((not judge, judge) if i < wins else
                (judge, not judge) if i < wins + losses else (True, True))
        identified_b += b
        identified_c += c
        record = {"case_id": row["case_id"], "status": "ok", "chat_type": row.get("chat_type")}
        record.update({"baseline_correct" if judge else "identified_baseline": b,
                       "candidate_correct" if judge else "identified_candidate": c})
        if b != c:
            vote_count = spec["protocol"]["flip_extra_rounds"] + 1
            verified = {"baseline_votes": [b] * vote_count, "candidate_votes": [c] * vote_count,
                        "baseline_identified_final": b, "candidate_identified_final": c}
            if judge:
                record.update(verified, flip_verified=True)
            else:
                record["flip_verified"] = verified
        records.append(record)
    if failures:
        for record in records[-failures:]:
            record.update(status="failed", reason="test failure")
    (directory / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    metrics = dict(pairs=len(rows) - failures, attempted=len(rows), failures=failures,
                   failure_rate=failures / len(rows), identified_baseline=identified_b,
                   identified_candidate=identified_c, wins_confirmed=wins,
                   losses_confirmed=losses, contested=0, net_win_confirmed=wins-losses)
    experiment.finish(directory, metrics, {"verdict": verdict, "reason": "test evidence"})
    return metrics
