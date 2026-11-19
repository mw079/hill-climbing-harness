#!/usr/bin/env python3
"""Read-only audit of a run produced by the current experiment contracts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hillclimb as hc  # noqa: E402


def audit(directory):
    run = hc.Run(directory)
    state, records = run.state, run.history
    errors = []
    benchmark_failures = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    require(len(records) == state["done"], "journal length differs from state.done")
    require([r["exp"] for r in records] == list(range(1, len(records) + 1)),
            "experiment sequence is not consecutive")
    require(state["status"] in ("finished", "converged"), "run has not completed")
    require(bool(state.get("final")), "final evaluation is missing")
    require(state["check"]["task_hash"] == hc.task_hash(run.task), "task changed since preflight")
    if state["status"] == "converged":
        require(bool(records) and records[-1]["decision"] == "stop", "convergence lacks stop record")

    logs = list((run.dir / "logs").glob("*.log"))
    main_sessions, review_sessions = set(), []
    for log in logs:
        text = log.read_text()
        require(hc.redact(text) == text, f"credential found in transcript {log.name}")
        reply = hc.read_json_text(text.split("\nSTDOUT\n", 1)[-1].split("\nSTDERR\n", 1)[0])
        session = reply.get("session_id")
        if not session or reply.get("is_error"):
            continue
        if any(f"-{step}.log" in log.name for step in ("review_plan", "review_code", "review_stop")):
            review_sessions.append(session)
        else:
            main_sessions.add(session)
    require(not main_sessions.intersection(review_sessions), "reviewer reused a main-agent session")
    require(len(review_sessions) == len(set(review_sessions)), "reviewer sessions were reused")
    if state["agent"] == "cursor":
        require(bool(main_sessions) and bool(review_sessions), "Cursor session evidence is missing")

    for rec in records:
        label = f"experiment {rec['exp']}"
        artifacts = run.dir / "experiments" / f"{rec['exp']:04d}"
        plan = artifacts / "plan.md"
        stop = artifacts / "stop.json"
        if plan.exists():
            require(hc.plan_problem(plan) is None, f"{label}: invalid archived plan")
            plan_text = plan.read_text()
        else:
            require(stop.exists(), f"{label}: missing plan or convergence request")
            plan_text = json.dumps(hc.read_json(stop, {}), indent=2)
        prefix = f"{rec['exp']:04d}-"
        analyses = [p.read_text() for p in logs if p.name.startswith(prefix) and p.name.endswith('-analyze.log')]
        require(any("<experiment_plan>\n" + plan_text + "\n</experiment_plan>" in t for t in analyses),
                f"{label}: analysis lacks its archived plan")
        require(hc.read_json(artifacts / "analysis.json") == rec["analysis"],
                f"{label}: analysis artifact differs from journal")
        if rec["sha"]:
            require(hc.git(run.ws, "rev-parse", rec["tag"]) == rec["sha"], f"{label}: tag differs from commit")
        if rec["decision"] == "keep":
            val = rec.get("val") or {}
            require(val.get("valid") is True, f"{label}: kept invalid candidate")
            gain = val.get("gain")
            require(gain is None or gain > state["check"]["threshold"], f"{label}: gain below threshold")
            reviews = hc.read_json(artifacts / "plan_review.json", [])
            require(bool(reviews) and reviews[-1]["verdict"] == "approve", f"{label}: plan not approved")
            reviews = hc.read_json(artifacts / "code_review.json", [])
            require(bool(reviews) and reviews[-1]["verdict"] == "approve", f"{label}: code not approved")
            tests = artifacts / "test.log"
            require(tests.exists() and ": pass\n" in tests.read_text(), f"{label}: no passing test evidence")
        if rec["decision"] == "stop":
            review = hc.read_json(artifacts / "stop_review.json", {})
            require(review.get("verdict") == "approve", f"{label}: stop not approved")

    kept = [r for r in records if r["decision"] == "keep"]
    if kept:
        require(state["best_sha"] == kept[-1]["sha"], "state best differs from last kept commit")
        if state["check"]["state"] == "scratch":
            require("first_valid" in (state.get("final") or {}), "scratch run lacks first-valid comparison")
            if kept[0]["sha"] == state["best_sha"]:
                require(state.get("final", {}).get("first_valid") == state.get("final", {}).get("best"),
                        "identical first/final candidate measurements differ")
    if state["check"]["baseline"] == "reference":
        require("reference" in (state.get("final") or {}), "hidden reference was not measured at final")
    require(hc.git(run.ws, "rev-parse", "best") == state["best_sha"], "best branch differs from state")
    require(not hc.git(run.ws, "status", "--porcelain", "--", ".", ":(exclude)research"),
            "workspace has uncommitted changes outside research")
    for name, result in (state.get("final") or {}).items():
        if name != "at":
            if result.get("valid") is True:
                require(len(result.get("scores", [])) == run.task["eval"]["repetitions"],
                        f"final {name}: incomplete measurement repetitions")
            else:
                # Invalid test results are legitimate observations, not corrupt
                # artifacts. The audit still fails, with the cause distinguished.
                require(result.get("median") is None and bool(result.get("error")),
                        f"final {name}: malformed invalid result")
                benchmark_failures.append(f"final {name}: {result.get('error')}")
    return {"run": str(run.dir), "ok": not errors and not benchmark_failures, "experiments": len(records),
            "decisions": [r["decision"] for r in records],
            "independent_review_sessions": len(review_sessions), "errors": errors,
            "benchmark_failures": benchmark_failures}


def audit_multi(directory):
    """Audit a hillclimb_multi (v2) run: records, champion line, measurements, sessions."""
    import hillclimb_multi as hm
    lab = hm.Lab(directory)
    st, recs = lab.state, lab.records
    errors, benchmark_failures = [], []

    def require(ok, message):
        if not ok:
            errors.append(message)

    exps = [r["exp"] for r in recs]
    require(len(exps) == len(set(exps)), "an experiment was recorded twice")
    require(sorted(exps) == list(range(1, len(exps) + 1)), "experiment numbers are not 1..N")
    require(set(st.get("claimed", [])) == set(exps), "claimed experiments differ from recorded ones")
    require(st["status"] in ("finished", "converged"), "run has not completed")
    require(bool(st.get("final")), "final evaluation is missing")
    notes = []
    if st["check"]["task_hash"] != hc.task_hash(lab.task):
        notes.append("the task folder has changed since this run's check (expected for older runs)")

    champion = {"sha": st["start_sha"]} if st["check"]["state"] == "existing" else None
    for r in recs:
        label, xd = f"experiment {r['exp']}", lab.dir / "experiments" / f"{r['exp']:04d}"
        if r["sha"]:
            require(hc.git(lab.repo, "rev-parse", r["tag"]) == r["sha"], f"{label}: tag differs from commit")
        if r["decision"] == "keep":
            v = r.get("val") or {}
            require(v.get("valid") is True, f"{label}: kept an invalid candidate")
            require(v.get("vs") == (champion or {}).get("sha"), f"{label}: not measured against the champion of its time")
            if champion is not None:
                require(v.get("gain") is not None and v["gain"] > st["check"]["threshold"], f"{label}: gain below threshold")
                require((r.get("confirm") or {}).get("decision") == "keep", f"{label}: champion not confirmed")
            reviews = hc.read_json(xd / "plan_review.json", [])
            require(bool(reviews) and reviews[-1]["verdict"] == "approve", f"{label}: plan not approved")
            reviews = hc.read_json(xd / "code_review.json", [])
            require(bool(reviews) and reviews[-1]["verdict"] == "approve", f"{label}: code not approved")
            tests = xd / "test.log"
            require(tests.exists() and ": pass\n" in tests.read_text(), f"{label}: no passing test evidence")
            champion = {"sha": r["sha"]}
        if r["decision"] == "stop":
            require(hc.read_json(xd / "stop_review.json", {}).get("verdict") == "approve", f"{label}: stop not approved")
    require((st.get("champion") or {}).get("sha") == (champion or {}).get("sha"), "state champion differs from records")
    if champion:
        require(hc.git(lab.repo, "rev-parse", "best") == champion["sha"], "best branch differs from the champion")

    lines = (lab.dir / "journal.jsonl").read_text().splitlines()
    meas = sorted((e for e in map(json.loads, lines) if e["event"] == "measure"), key=lambda e: e["start"])
    require(all(a["end"] <= b["start"] + 1e-6 for a, b in zip(meas, meas[1:])), "two measurements overlapped")

    worker_sessions, review_sessions = set(), []
    for log in (lab.dir / "logs").glob("*.log"):
        text = log.read_text()
        require(hc.redact(text) == text, f"credential found in transcript {log.name}")
        reply = hc.read_json_text(text.split("\nSTDOUT\n", 1)[-1].split("\nSTDERR\n", 1)[0])
        session = reply.get("session_id")
        if not session or reply.get("is_error"):
            continue
        if any(f"-{s}.log" in log.name for s in ("review_plan", "review_code", "review_stop")):
            review_sessions.append(session)
        else:
            worker_sessions.add(session)
    require(not worker_sessions.intersection(review_sessions), "a reviewer reused a worker session")
    require(len(review_sessions) == len(set(review_sessions)), "reviewer sessions were reused")
    for name, result in (st.get("final") or {}).items():
        if name != "at" and result.get("valid") is not True:
            benchmark_failures.append(f"final {name}: {result.get('error')}")
    return {"run": str(lab.dir), "ok": not errors and not benchmark_failures, "experiments": len(recs),
            "decisions": [r["decision"] for r in recs], "measurements": len(meas),
            "independent_review_sessions": len(review_sessions), "errors": errors, "notes": notes,
            "benchmark_failures": benchmark_failures}


def audit_agent(directory):
    """Audit a hillclimb_agent (v3) run. Read-only: v3's own loader would record unfinished
    experiments as abandoned, so the run is read with the plain lab loader instead.

    The lead promotes; Python must refuse anything the measurements do not support. So the
    promotions are replayed in order: each promoted candidate's latest measurement (its own, or
    a later `./lab measure`) must be a confirmed win over the champion of that moment.
    """
    import hillclimb_agent as ha
    import hillclimb_multi as hm

    class ReadOnlyLab(hm.Lab):
        HARNESS = "agent"

    lab = ReadOnlyLab(directory)
    st, recs = lab.state, lab.records
    errors, notes, benchmark_failures = [], [], []

    def require(ok, message):
        if not ok:
            errors.append(message)

    exps = [r["exp"] for r in recs]
    require(len(exps) == len(set(exps)), "an experiment was recorded twice")
    require(sorted(exps) == list(range(1, len(exps) + 1)), "experiment numbers are not 1..N")
    require(set(st.get("claimed", [])) == set(exps), "claimed experiments differ from recorded ones")
    require(st["status"] in ("finished", "converged"), "run has not completed")
    require(bool(st.get("final")), "final evaluation is missing")
    if st["status"] == "converged":
        require(bool(st.get("stop_reason")), "converged without a recorded stop reason")
    if st["check"]["task_hash"] != hc.task_hash(lab.task):
        notes.append("the task folder has changed since this run's check (expected for older runs)")

    by_exp = {r["exp"]: r for r in recs}
    for r in recs:
        label = f"experiment {r['exp']}"
        if r["sha"]:
            require(hc.git(lab.repo, "rev-parse", r["tag"]) == r["sha"], f"{label}: tag differs from commit")
        if r["decision"] == "abandoned":
            require(not r.get("sha") and not r.get("val"), f"{label}: abandoned but has code or a measurement")
        if r.get("eligible"):
            require(r["decision"] == "keep", f"{label}: eligible without a confirmed win")

    journal = [json.loads(x) for x in (lab.dir / "journal.jsonl").read_text().splitlines() if x.strip()]
    promotions = st.get("promotions", [])
    require([p["exp"] for p in promotions] == [e["exp"] for e in journal if e["event"] == "promote"],
            "promotions in state differ from the journal")
    champion = st["start_sha"] if st["check"]["state"] == "existing" else None
    remeasured = st.get("remeasures", {})
    for p in promotions:
        r = by_exp.get(p["exp"])
        label = f"promotion of experiment {p['exp']}"
        if r is None:
            errors.append(f"{label}: no such experiment")
            continue
        require(r["decision"] not in ha.NOT_PROMOTABLE and r.get("sha"), f"{label}: {r['decision']} cannot be promoted")
        own = r.get("val") or {}
        again = remeasured.get(str(r["exp"]), {})
        supported = ((own.get("vs") == champion and r["decision"] == "keep")
                     or (again.get("vs") == champion and again.get("decision") == "keep"))
        require(supported, f"{label}: no confirmed win over the champion of that moment")
        if champion is not None and own.get("vs") == champion and r["decision"] == "keep":
            require(own.get("gain") is not None and own["gain"] > st["check"]["threshold"], f"{label}: gain below threshold")
            require((r.get("confirm") or {}).get("decision") == "keep", f"{label}: win not confirmed")
        xd = lab.dir / "experiments" / f"{r['exp']:04d}"
        reviews = hc.read_json(xd / "plan_review.json", [])
        require(bool(reviews) and reviews[-1]["verdict"] == "approve", f"{label}: plan not approved")
        reviews = hc.read_json(xd / "code_review.json", [])
        require(bool(reviews) and reviews[-1]["verdict"] == "approve"
                and reviews[-1].get("audit") != "flagged", f"{label}: code not approved, or flagged by the audit")
        tests = xd / "test.log"
        require(tests.exists() and ": pass\n" in tests.read_text(), f"{label}: no passing test evidence")
        champion = r["sha"]
    require((st.get("champion") or {}).get("sha") == champion, "state champion differs from the last promotion")
    if champion:
        require(hc.git(lab.repo, "rev-parse", "best") == champion, "best branch differs from the champion")
    if champion and champion != st["start_sha"]:
        fa = st.get("final_audit") or {}
        require(fa.get("audit") in ("clean", "flagged"), "the champion has no final audit verdict")
        if fa.get("audit") == "flagged":
            benchmark_failures.append("final audit flagged the champion: " + "; ".join(fa.get("issues", [])))

    meas = sorted((e for e in journal if e["event"] == "measure"), key=lambda e: e["start"])
    require(all(a["end"] <= b["start"] + 1e-6 for a, b in zip(meas, meas[1:])), "two measurements overlapped")

    worker, lead, review = set(), set(), []
    for log in (lab.dir / "logs").glob("*.log"):
        text = log.read_text()
        require(hc.redact(text) == text, f"credential found in transcript {log.name}")
        reply = hc.read_json_text(text.split("\nSTDOUT\n", 1)[-1].split("\nSTDERR\n", 1)[0])
        session = reply.get("session_id")
        if not session or reply.get("is_error"):
            continue
        if any(log.name.endswith(f"-{s}.log") for s in ("review_plan", "review_code", "review_stop", "final_audit")):
            review.append(session)
        elif log.name.endswith("-lead.log"):
            lead.add(session)
        else:
            worker.add(session)
    require(not (worker | lead).intersection(review), "a reviewer reused a worker or lead session")
    require(len(review) == len(set(review)), "reviewer sessions were reused")
    require(not lead.intersection(worker), "the lead shared a session with a worker")
    for name, result in (st.get("final") or {}).items():
        if name != "at" and result.get("valid") is not True:
            benchmark_failures.append(f"final {name}: {result.get('error')}")
    return {"run": str(lab.dir), "ok": not errors and not benchmark_failures, "experiments": len(recs),
            "decisions": [r["decision"] for r in recs], "promotions": [p["exp"] for p in promotions],
            "measurements": len(meas), "lead_sessions": len(lead), "independent_review_sessions": len(review),
            "errors": errors, "notes": notes, "benchmark_failures": benchmark_failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    hc.load_dotenv(hc.HARNESS / ".env")
    harness = (hc.read_json(args.run / "state.json") or {}).get("harness")
    try:
        result = {"multi": audit_multi, "agent": audit_agent}.get(harness, audit)(args.run)
    except hc.Fail as e:                                 # e.g. the run's task folder no longer exists
        raise SystemExit(f"cannot audit {args.run}: {e}")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
