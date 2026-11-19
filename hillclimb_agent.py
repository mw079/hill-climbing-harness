#!/usr/bin/env python3
"""Agent-led hill-climbing harness (v3): a lead agent runs the lab through a `lab` tool.

    python hillclimb_agent.py run    tasks/<name> --agent claude [--lead-agent X] [--workers 3]
    python hillclimb_agent.py resume [run]
    python hillclimb_agent.py status [run]
    python hillclimb_agent.py report [run]

The lead decides what to try, from which parent, when to merge, what to promote and when to
stop. Python (the lab from hillclimb_multi.py) starts every worker session, measures every
candidate one at a time, refuses promotions the measurements do not support, enforces the
limits and hides the hidden data, then audits the champion and runs the test split once.
Workers are the same sessions as in hillclimb_multi.py. See README.md.
"""
import argparse
import json
import shlex
import string
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import hillclimb as hc
import hillclimb_multi as hm
from hillclimb import AgentDown, ContractError, Fail, log, num, read_json

HARNESS = hc.HARNESS
WAIT_MAX, MAX_SEGMENTS = 540, 40
NOT_PROMOTABLE = ("flagged", "invalid", "guard-failed", "rejected", "stop", "stop-rejected", "abandoned")

LEAD = """[hillclimb step=lead exp=0]

You are the research lead of an automated lab that improves a piece of software. You decide
what the lab tries; you do not write code yourself. Up to $workers worker agents run at the
same time, each carrying out one idea in its own copy of the code. Every experiment passes a
plan review, a code review with an audit for benchmark gaming, the tests, and a measurement
against the current champion on a hidden validation split. Nobody is watching: never ask
questions, and keep working until the research is done.

Your workspace holds `TASK.md` (read it first) and the code of the champion at the time your
session started. You may read code and git history: `git tag` lists every candidate and
`git show <tag>` shows one. Do not edit code. Do not read files outside this workspace; use
the lab tool instead.

The lab tool (run it with your shell, from this directory):

    ./lab status                      champion, running workers, every candidate, limits left
    ./lab start-worker --parent champion|start|<exp> --idea "<what to try, and why>"
                                      start one experiment; refused when $workers workers are
                                      busy or a limit is reached
    ./lab start-worker --parent champion --merge-from <exp> --idea "<how to combine>"
                                      port another candidate's change onto a parent
    ./lab wait                        block until a running experiment finishes (up to 9
                                      minutes; call again if still running); prints results
    ./lab measure <exp>               re-measure a candidate against the current champion
    ./lab promote <exp>               make it the champion; refused unless its latest
                                      measurement beat the current champion, was confirmed,
                                      and passed the audit and the guard check
    ./lab notebook                    every finished experiment with the worker's lesson
    ./lab note "<text>"               add to your notes (your memory: context gets compacted)
    ./lab notes                       read your notes
    ./lab agenda-write "<text>"       the agenda every worker reads when planning
    ./lab stop "<reason>"             end the research; running experiments finish first

$goal

Limits: at most $max_experiments experiments and $max_hours hours. After you stop, the
controller audits the champion and evaluates it once on the hidden test split.

How to run the lab well:

- Keep the workers busy with distinct ideas in different parts of the system. Parallel workers
  that try the same thing waste experiments. While workers run, use `./lab wait`; do not end
  your turn while experiments are running.
- Prefer fewer, deeper, well-reasoned experiments over many small tweaks. Each idea should name
  the bottleneck it attacks and the evidence for it.
- Promote only what the measurements support. With many candidates, lucky measurements happen;
  re-measure surprises. A candidate measured against an older champion must be re-measured
  before it can be promoted.
- A candidate that beat an older parent but not the newer champion may be worth a merge.
- Generalisation matters: the test split may contain data unlike train and validation.
- Write your plan and what you learn with `./lab note` regularly.
- Stop with a reason when the evidence says no worthwhile experiment remains, or let the limits
  end the run. Do not stop early just to finish.

Start with `./lab status`."""

CONTINUE = """[hillclimb step=lead exp=0]

You are the research lead of this lab; your instructions are earlier in this session. Your
previous turn ended. The lab now looks like this:

$status

Continue: start workers, `./lab wait` for results, promote, merge, or `./lab stop "<reason>"`."""

FINAL_AUDIT = """[hillclimb step=final_audit exp=0]

You audit the final result of an automated research lab. You did not write it. The change from
the starting code to the final champion is:

    git diff $start HEAD -- . ':(exclude)research'

Read `TASK.md`. Flag the change if it special-cases the benchmark (detects inputs, sizes or
values; hard-codes answers or constants that only fit the training data), caches or reuses
results across calls, interferes with timing or correctness checks, lowers precision or
accuracy below what TASK.md allows, uses forbidden libraries or copied source, or reads files
it may not read.

Write only `research/review.json`, valid JSON:

```
{"audit": "clean" | "flagged", "audit_issues": ["specific concern", "..."]}
```

A flagged audit needs at least one issue. Change nothing else."""


class AgentLab(hm.Lab):
    HARNESS = "agent"
    auto_promote = False

    def __init__(self, run_dir):
        super().__init__(run_dir)
        agents = tomllib.loads(Path(self.state["agents_file"]).read_text())
        self.lead_cfg = agents[self.state["lead_agent"]]
        self.running, self.unread = {}, []                # slot -> experiment info; finished, not yet reported
        self.done_cv = threading.Condition(self.lock)
        self.server_stop = threading.Event()
        self.threads = []
        # Experiments claimed in an earlier session but never recorded are abandoned: the lead
        # decides afresh what to run. They stay in the record so numbering stays complete.
        for n in self.pending:
            self.record({"exp": n, "slot": 0, "parent": None, "decision": "abandoned", "title": "abandoned",
                         "reason": "the run stopped before this experiment finished", "hypothesis": "",
                         "area": None, "val": None, "sha": None, "tag": None, "seconds": 0})
        self.pending = []

    def settings_for(self, slot):
        s = super().settings_for(slot)
        if slot == 0:                                     # the lead may read its own workspace
            lead = str(self.dir / "worktrees" / "lead")
            s["permissions"]["deny"] = [r for r in s["permissions"]["deny"] if lead not in r]
            fs = s["sandbox"]["filesystem"]
            fs["denyRead"] = [p for p in fs["denyRead"] if p != lead]
        return s

    def champion_history(self):
        """Records that became champion: only what the lead promoted, in promotion order."""
        by_exp = {r["exp"]: r for r in self.records}
        return [by_exp[p["exp"]] for p in self.state.get("promotions", []) if p["exp"] in by_exp]

    def limits_reached(self):
        return (len(self.state.get("claimed", [])) >= self.limits["max_experiments"]
                or self.elapsed() >= self.limits["max_hours"] * 3600)

    def stopping(self):
        return bool(self.state.get("stop_reason"))

    # ---- lab tool: server side

    def serve(self):
        requests = self.dir / "lab" / "requests"
        while not self.server_stop.is_set():
            for f in sorted(requests.glob("*.json"), key=lambda p: p.stat().st_mtime):
                data = read_json(f) or {}
                f.unlink(missing_ok=True)
                threading.Thread(target=self.answer, args=(f.name, data.get("args", [])), daemon=True).start()
            time.sleep(0.2)

    def answer(self, name, args):
        try:
            code, text = self.command([str(a) for a in args])
        except Fail as e:
            code, text = 1, f"error: {e}"
        except Exception:
            code, text = 1, "internal error in the lab tool:\n" + traceback.format_exc()[-1500:]
        self.journal("lab", args=args[:2], code=code)
        out = self.dir / "lab" / "results" / name
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"code": code, "text": text}))
        tmp.replace(out)

    def command(self, args):
        if not args:
            return 1, "usage: ./lab status|start-worker|wait|measure|promote|notebook|note|notes|agenda|agenda-write|stop"
        cmd, rest = args[0], args[1:]
        handlers = {"status": lambda: (0, self.status_text()), "wait": lambda: self.cmd_wait(rest),
                    "start-worker": lambda: self.cmd_start(rest), "measure": lambda: self.cmd_measure(rest),
                    "promote": lambda: self.cmd_promote(rest), "notebook": lambda: (0, self.notebook_text()),
                    "note": lambda: self.cmd_note(rest), "notes": lambda: (0, self.read("lead_notes.md")),
                    "agenda": lambda: (0, self.read("agenda.md")), "agenda-write": lambda: self.cmd_agenda(rest),
                    "stop": lambda: self.cmd_stop(rest)}
        if cmd not in handlers:
            return 1, f"unknown lab command '{cmd}'"
        return handlers[cmd]()

    def read(self, name):
        f = self.dir / name
        return f.read_text() if f.exists() else "(empty)"

    def option(self, rest, flag):
        if flag in rest and rest.index(flag) + 1 < len(rest):
            return rest[rest.index(flag) + 1]
        return None

    def find(self, exp):
        try:
            n = int(str(exp).lstrip("exp-").lstrip("#"))
        except ValueError:
            raise Fail(f"'{exp}' is not an experiment number")
        with self.lock:
            rec = next((r for r in self.records if r["exp"] == n), None)
        if rec is None:
            raise Fail(f"experiment {n} has no result yet")
        return rec

    def parent_of(self, spec):
        with self.lock:
            champion = self.champion
        if spec in (None, "champion"):
            return champion or {"sha": self.state["start_sha"], "exp": 0, "score": None}
        if spec == "start":
            return {"sha": self.state["start_sha"], "exp": 0,
                    "score": self.state["check"]["target_val"] if self.state["check"]["state"] == "existing" else None}
        rec = self.find(spec)
        if not rec.get("sha") or rec["decision"] in NOT_PROMOTABLE:
            raise Fail(f"experiment {rec['exp']} ({rec['decision']}) cannot be a parent")
        return {"sha": rec["sha"], "exp": rec["exp"], "score": (rec.get("val") or {}).get("score")}

    def cmd_start(self, rest):
        idea = self.option(rest, "--idea")
        if not idea or not idea.strip():
            return 1, "start-worker needs --idea \"<what to try, and why>\""
        parent = self.parent_of(self.option(rest, "--parent"))
        merge = None
        if self.option(rest, "--merge-from"):
            m = self.find(self.option(rest, "--merge-from"))
            if not m.get("sha"):
                return 1, f"experiment {m['exp']} has no code to merge"
            merge = {k: m.get(k) for k in ("exp", "tag", "parent", "area", "title")}
        with self.lock:
            free = [k for k in range(1, self.workers + 1) if k not in self.running]
            if not free:
                return 1, f"refused: all {self.workers} workers are busy; use ./lab wait"
            if self.stopping():
                return 1, "refused: the research was stopped"
            n = self.claim()
            if n is None:
                return 1, "refused: the experiment or time limit is reached"
            slot = free[0]
            self.running[slot] = {"exp": n, "idea": idea, "parent": parent.get("exp"), "since": time.time()}
        t = threading.Thread(target=self.run_experiment, args=(slot, n, parent, idea, merge), name=f"w{slot}",
                             daemon=True)
        self.threads.append(t)
        t.start()
        return 0, f"started experiment {n} on worker {slot} from {'the start' if not parent.get('exp') else 'experiment ' + str(parent['exp'])}"

    def run_experiment(self, slot, n, parent, idea, merge):
        try:
            self.experiment(slot, n, parent=parent, idea=idea, merge=merge)
        except hm.Paused:
            pass
        except (AgentDown, ContractError) as e:
            self.pause(str(e))
        except Exception:
            self.fatal = traceback.format_exc()
            self.pause(f"internal error in worker {slot}")
        finally:
            with self.done_cv:
                self.running.pop(slot, None)
                self.unread.append(n)
                self.done_cv.notify_all()

    def cmd_wait(self, rest):
        timeout = min(float(self.option(rest, "--timeout") or WAIT_MAX), WAIT_MAX)
        deadline = time.time() + timeout
        with self.done_cv:
            while not self.unread and self.running and time.time() < deadline and not self.abort.is_set():
                self.done_cv.wait(min(5, max(0.1, deadline - time.time())))
            finished, self.unread = self.unread, []
            running = dict(self.running)
        if not finished:
            if running:
                return 0, "still running: " + ", ".join(f"experiment {r['exp']}" for r in running.values()) + \
                    ". Call ./lab wait again."
            return 0, "no worker is running and no result is waiting."
        return 0, "\n\n".join(self.result_text(n) for n in finished)

    def result_text(self, n):
        with self.lock:
            rec = next((r for r in self.records if r["exp"] == n), None)
            note = self.notes.get(n, {})
        if rec is None:
            return f"experiment {n}: not recorded (the lab is pausing)"
        v = rec.get("val") or {}
        lines = [f"experiment {n} (worker {rec['slot']}): {rec['decision'].upper()} - {rec['title']}",
                 f"  reason: {rec['reason']}"]
        if v:
            lines.append(f"  val score {num(v.get('score'))} vs champion {num(v.get('best_score'))} "
                         f"(gain {num(v.get('gain'))}, {v.get('pairs')} pairs, measured against "
                         f"{'experiment ' + str(rec.get('vs_exp')) if rec.get('vs_exp') else 'the start'})")
        if rec.get("eligible"):
            lines.append(f"  eligible for promotion: ./lab promote {n}")
        if note:
            lines.append(f"  lesson: {note.get('lesson')}\n  suggested next: {note.get('next')}")
        return "\n".join(lines)

    def latest(self, rec):
        """The candidate's most recent measurement: its own, or a later ./lab measure."""
        again = self.state.get("remeasures", {}).get(str(rec["exp"]))
        if again:
            return again
        v = rec.get("val") or {}
        return {"vs": v.get("vs"), "decision": rec["decision"], "score": v.get("score"), "gain": v.get("gain")}

    def cmd_measure(self, rest):
        if not rest:
            return 1, "usage: ./lab measure <exp>"
        rec = self.find(rest[0])
        if not rec.get("sha") or rec["decision"] in NOT_PROMOTABLE:
            return 1, f"experiment {rec['exp']} ({rec['decision']}) cannot be measured"
        r = self.submit("remeasure", sha=rec["sha"], exp=rec["exp"])
        return 0, (f"experiment {rec['exp']} re-measured against the current champion: {r['decision']}, "
                   f"val score {num(r.get('score'))} vs {num(r.get('best_score'))} (gain {num(r.get('gain'))})")

    def measure_remeasure(self, sha, exp):
        with self.lock:
            champion = self.champion
        v = self.compare(sha, champion)
        if v["decision"] == "keep" and champion is not None:
            confirm = self.compare(sha, champion)["decision"]
            if confirm == "invalid":
                v["decision"] = "invalid"
            elif confirm != "keep":
                v["decision"] = "keep-unconfirmed"
        if v["decision"] in hm.VALUED and self.task["p"]["guard"]:
            if not self.evaluate(self.task, self.dir / "worktrees" / "m-cand", self.task["p"]["guard"])["valid"]:
                v["decision"] = "guard-failed"
        out = {"vs": champion["sha"] if champion else None, "decision": v["decision"], "score": v.get("score"),
               "best_score": v.get("best_score"), "gain": v.get("gain"), "at": time.time()}
        with self.lock:
            self.state.setdefault("remeasures", {})[str(exp)] = out
            self.save()
        return out

    def cmd_promote(self, rest):
        if not rest:
            return 1, "usage: ./lab promote <exp>"
        rec = self.find(rest[0])
        with self.lock:
            champion = self.champion
            if rec["decision"] in NOT_PROMOTABLE or not rec.get("sha"):
                return 1, f"refused: experiment {rec['exp']} is {rec['decision']}"
            if champion and rec["sha"] == champion["sha"]:
                return 0, f"experiment {rec['exp']} is already the champion"
            m = self.latest(rec)
            if m["vs"] != (champion or {}).get("sha"):
                return 1, (f"refused: experiment {rec['exp']} was measured against an older champion; "
                           f"run ./lab measure {rec['exp']} first")
            if m["decision"] != "keep":
                return 1, (f"refused: its latest measurement ({m['decision']}, gain {num(m.get('gain'))}) does not "
                           f"show a confirmed win over the current champion")
            self.state["champion"] = {"sha": rec["sha"], "score": m["score"], "exp": rec["exp"]}
            self.state.setdefault("promotions", []).append({"exp": rec["exp"], "at": time.strftime("%H:%M:%S"),
                                                            "previous": (champion or {}).get("exp")})
            self.g(self.repo, "branch", "-f", "best", rec["sha"])
            self.save()
        self.journal("promote", exp=rec["exp"])
        log(f"  [lead] promoted experiment {rec['exp']} to champion")
        return 0, f"experiment {rec['exp']} is now the champion (val score {num(m['score'])})"

    def cmd_note(self, rest):
        text = " ".join(rest).strip()
        if not text:
            return 1, "usage: ./lab note \"<text>\""
        with self.lock, open(self.dir / "lead_notes.md", "a") as f:
            f.write(f"- [{time.strftime('%H:%M')}] {text}\n")
        return 0, "noted"

    def cmd_agenda(self, rest):
        text = " ".join(rest).strip()
        if not text:
            return 1, "usage: ./lab agenda-write \"<text>\""
        (self.dir / "agenda.md").write_text(text + "\n")
        return 0, "agenda written; workers read it when they plan"

    def cmd_stop(self, rest):
        reason = " ".join(rest).strip()
        if not reason:
            return 1, "usage: ./lab stop \"<reason>\""
        with self.lock:
            self.state["stop_reason"] = reason
            self.save()
        return 0, "research stopped; running experiments will finish"

    def notebook_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            hc.write_history(SimpleNamespace(history=self.history_view(), ws=Path(tmp)))
            return (Path(tmp) / "research" / "HISTORY.md").read_text()

    def status_text(self):
        with self.lock:
            st, c = self.state, self.champion
            lines = [f"champion: {('experiment ' + str(c['exp'])) if c and c['exp'] else ('the start' if c else 'none')}"
                     f", val score {num(c['score']) if c else 'n/a'}"]
            if st["check"]["baseline"] == "reference":
                lines.append(f"reference to beat: val score {num(st['check']['scores'].get('val'))}")
            lines.append(f"experiments started: {len(st.get('claimed', []))} of {self.limits['max_experiments']}; "
                         f"hours: {self.elapsed() / 3600:.2f} of {self.limits['max_hours']}")
            if st.get("stop_reason"):
                lines.append(f"research stopped: {st['stop_reason']}")
            lines.append("running: " + ("; ".join(f"worker {k}: experiment {r['exp']} from "
                         f"{'start' if not r['parent'] else 'exp ' + str(r['parent'])}, {r['idea'][:80]}"
                         for k, r in sorted(self.running.items())) or "nothing"))
            if self.unread:
                lines.append(f"finished, not yet read with ./lab wait: {sorted(self.unread)}")
            lines.append("candidates:")
            for r in self.records:
                v = r.get("val") or {}
                m = self.latest(r)
                flag = " ELIGIBLE" if m["decision"] == "keep" and m["vs"] == (c or {}).get("sha") \
                    and r["decision"] not in NOT_PROMOTABLE and (not c or r.get("sha") != c["sha"]) else ""
                lines.append(f"  {r['exp']}: {r['decision']}{flag}, {r['title'][:70]}, val {num(v.get('score'))}, "
                             f"tag {r.get('tag') or '-'}")
            for p in st.get("promotions", []):
                lines.append(f"promoted: experiment {p['exp']} at {p['at']}")
        return "\n".join(lines)


hm.PRIORITY["remeasure"] = 0


# ---------------------------------------------------------------- the lead

def lab_script(lab):
    return ("#!/bin/sh\n# The research lead's tool. See ./lab status.\n"
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(HARNESS / 'hillclimb_agent.py'))} "
            f"lab-call {shlex.quote(str(lab.dir))} \"$@\"\n")


def progress(lab):
    with lab.lock:
        return (len(lab.state.get("claimed", [])), len(lab.records), len(lab.state.get("promotions", [])),
                lab.state.get("stop_reason"), len(lab.state.get("remeasures", {})))


def lead_loop(lab):
    wt = lab.dir / "worktrees" / "lead"
    idle = 0
    while not lab.abort.is_set():
        with lab.lock:
            busy = bool(lab.running)
        if (lab.stopping() or lab.limits_reached()) and not busy:
            break
        if lab.state.get("lead_segments", 0) >= MAX_SEGMENTS:
            lab.state["stop_reason"] = f"the lead's session ended {MAX_SEGMENTS} times"
            break
        with lab.lock:
            champion = lab.champion or {"sha": lab.state["start_sha"]}
        if not busy:
            lab.prepare(wt, champion["sha"])
        (wt / "lab").write_text(lab_script(lab))
        (wt / "lab").chmod(0o755)
        values = {"workers": lab.workers, "goal": lab.goal(), "max_experiments": lab.limits["max_experiments"],
                  "max_hours": lab.limits["max_hours"], "status": lab.status_text()}
        first = not lab.state.get("lead_session")
        text = string.Template(LEAD if first else CONTINUE).substitute(values)
        before = progress(lab)
        remaining = lab.limits["max_hours"] * 3600 - lab.elapsed()
        _, session = lab.call_agent(0, 0, "lead", wt, {}, lab.lead_cfg, session=lab.state.get("lead_session"),
                                    raw_text=text, timeout=max(600, int(remaining) + 600))
        with lab.lock:
            lab.state["lead_session"] = session or lab.state.get("lead_session")
            lab.state["lead_segments"] = lab.state.get("lead_segments", 0) + 1
            lab.save()
        with lab.done_cv:                                 # give the lead news before its next turn
            while lab.running and not lab.unread and not lab.abort.is_set():
                lab.done_cv.wait(5)
            busy = bool(lab.running)
        idle = idle + 1 if progress(lab) == before and not busy and not lab.unread else 0
        if idle >= 2:
            lab.state["stop_reason"] = "the lead ended twice in a row without starting, promoting or stopping anything"
            break
    for t in lab.threads:
        t.join()


def final_audit(lab):
    """An independent session (not the lead's) audits the champion before the test split."""
    c = lab.champion
    if not c or c["sha"] == lab.state["start_sha"]:
        return None
    wt = lab.dir / "worktrees" / "lead"
    lab.prepare(wt, c["sha"])
    f = wt / "research" / "review.json"
    f.unlink(missing_ok=True)
    text = FINAL_AUDIT.replace("$start", lab.state["start_sha"])
    lab.call_agent(0, 0, "final_audit", wt, {}, lab.review_cfg, fresh=True, raw_text=text)
    v = read_json(f) or {}
    result = {"audit": v.get("audit") if v.get("audit") in ("clean", "flagged") else "no verdict",
              "issues": v.get("audit_issues", []), "exp": c["exp"]}
    lab.state["final_audit"] = result
    lab.save()
    return result


def start(lab):
    st = lab.state
    if st.get("final"):
        raise Fail("this run already used the test split; no more experiments are allowed")
    st["status"] = "running"
    lab.save()
    log(f"run {st['run_id']}: lead={st['lead_agent']}, up to {lab.workers} workers ({st['agent']}), "
        f"reviews={st['review_agent']}, limit {lab.limits['max_experiments']} experiments or "
        f"{lab.limits['max_hours']} hours")
    for kind in ("dev", "eval") + (("ref",) if lab.task["p"]["reference"] else ()):
        hc.env_for(lab.task, kind)
    meter = threading.Thread(target=lab.measurement_loop, name="measure", daemon=True)
    server = threading.Thread(target=lab.serve, name="lab", daemon=True)
    meter.start()
    server.start()
    try:
        lead_loop(lab)
    except hm.Paused:
        pass
    except (AgentDown, ContractError) as e:
        lab.pause(str(e))
    except KeyboardInterrupt:
        lab.pause("interrupted")
    finally:
        for t in lab.threads:
            t.join()
        lab.measure_stop.set()
        meter.join()
        lab.server_stop.set()
        server.join()
    if lab.fatal:
        st["status"] = "paused"
        lab.save()
        raise Fail(f"internal error; run paused\n{lab.fatal}")
    if lab.pause_reason:
        st["status"] = "paused"
        lab.save()
        report(lab)
        raise Fail(f"{lab.pause_reason}\n\nRun paused; unfinished experiments will be recorded as abandoned and "
                   f"the test split is untouched. Continue with: python hillclimb_agent.py resume {lab.dir}")
    st["status"] = "converged" if st.get("stop_reason") else "finished"
    lab.save()
    final_audit(lab)
    if lab.task["p"]["test"]:
        hm.final(lab)
    print()
    print(report(lab))
    print(f"report: {lab.dir / 'report.md'}")


def report(lab):
    text = hm.report(lab).replace("(hillclimb_multi)", "(hillclimb_agent)").replace(
        "new champion(s)", "eligible win(s)")
    st = lab.state
    extra = ["", "## The lead", "",
             f"- Lead agent: {st['lead_agent']}; sessions: {st.get('lead_segments', 0)}",
             f"- Stop reason: {st.get('stop_reason') or 'limits reached'}"]
    extra += [f"- Promoted experiment {p['exp']} at {p['at']} (previous champion: "
              f"{'start' if not p['previous'] else 'experiment ' + str(p['previous'])})"
              for p in st.get("promotions", [])] or ["- No promotion"]
    fa = st.get("final_audit")
    if fa:
        extra.append(f"- Final independent audit of experiment {fa['exp']}: {fa['audit']}"
                     + (f" ({'; '.join(fa['issues'])})" if fa["issues"] else ""))
    if (lab.dir / "lead_notes.md").exists():
        extra += ["", "### Lead notes", "", (lab.dir / "lead_notes.md").read_text().strip()]
    text = text.replace("\n## Inspecting the run", "\n".join(extra) + "\n\n## Inspecting the run", 1)
    (lab.dir / "report.md").write_text(text)
    return text


def lab_call(run_dir, args):
    """Client side of ./lab: send one command to the lab and print its answer."""
    lab_dir = Path(run_dir) / "lab"
    rid = uuid.uuid4().hex
    tmp = lab_dir / "requests" / f"{rid}.tmp"
    tmp.write_text(json.dumps({"args": args}))
    tmp.replace(lab_dir / "requests" / f"{rid}.json")
    reply = lab_dir / "results" / f"{rid}.json"
    deadline = time.time() + WAIT_MAX + 3600
    while not reply.exists():
        if time.time() > deadline:
            print("error: no answer from the lab (is the run still going?)")
            return 1
        time.sleep(0.3)
    r = json.loads(reply.read_text())
    print(r["text"])
    try:
        reply.unlink()                                    # the sandbox may forbid it; the lab cleans up
    except OSError:
        pass
    return r["code"]


def find_run(arg, runs_dir):
    if arg:
        for cand in (Path(arg), Path(runs_dir) / arg):
            if (cand / "state.json").exists():
                return AgentLab(cand)
        raise Fail(f"no run found at '{arg}'")
    runs = sorted((d for d in Path(runs_dir).glob("*-agent-*") if (d / "state.json").exists()),
                  key=lambda d: (d / "state.json").stat().st_mtime)
    if not runs:
        raise Fail(f"no hillclimb_agent runs in {runs_dir}")
    return AgentLab(runs[-1])


def main(argv=None):
    hc.load_dotenv(HARNESS / ".env")
    ap = argparse.ArgumentParser(description="Agent-led hill-climbing harness (v3)")
    ap.add_argument("--runs-dir", default=str(hc.HOME / "runs"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--agent", default="claude", help="agents.toml entry for workers")
    r.add_argument("--lead-agent", help="agents.toml entry for the lead (default: --agent)")
    r.add_argument("--review-agent", help="agents.toml entry for reviews (default: --agent)")
    r.add_argument("--agents-file", default=str(HARNESS / "agents.toml"))
    r.add_argument("--workers", type=int, default=3)
    resume = sub.add_parser("resume")
    resume.add_argument("run", nargs="?")
    for p in (r, resume):
        p.add_argument("--max-experiments", type=int)
        p.add_argument("--max-hours", type=float)
    for name in ("status", "report"):
        sub.add_parser(name).add_argument("run", nargs="?")
    lc = sub.add_parser("lab-call")
    lc.add_argument("run_dir")
    lc.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    limits = {k: v for k, v in (("max_experiments", getattr(a, "max_experiments", None)),
                                ("max_hours", getattr(a, "max_hours", None))) if v is not None}
    if a.cmd == "lab-call":
        return lab_call(a.run_dir, a.args)
    if a.cmd == "run":
        task = hc.load_task(a.task)
        rep = read_json(task["dir"] / "check_report.json")
        if task["problems"] or not rep or not rep["ok"] or rep["task_hash"] != hc.task_hash(task):
            raise Fail(f"the task has no passing, current check. Run: python hillclimb.py check {a.task}")
        agents_file = Path(a.agents_file).resolve()
        agents = tomllib.loads(agents_file.read_text())
        lead_agent, review_agent = a.lead_agent or a.agent, a.review_agent or a.agent
        for name in (a.agent, lead_agent, review_agent):
            if name not in agents:
                raise Fail(f"agent '{name}' is not defined in {agents_file}")
        if a.workers < 1:
            raise Fail("--workers must be at least 1")
        limits = {"max_experiments": task["budget"]["max_experiments"], "max_hours": task["budget"]["max_hours"],
                  **limits, "workers": a.workers}
        start(_new_agent_run(task, rep, a.agent, lead_agent, review_agent, agents_file, a.runs_dir, limits))
        return 0
    lab = find_run(a.run, a.runs_dir)
    if a.cmd == "resume":
        lab.state["limits"].update(limits)
        lab.limits.update(limits)
        start(lab)
    elif a.cmd == "report":
        print(report(lab))
    elif a.cmd == "status":
        print(lab.status_text())
    return 0


def _new_agent_run(task, rep, agent, lead_agent, review_agent, agents_file, runs_dir, limits):
    class Bootstrap(hm.Lab):                              # reads the fresh run before lead_agent is set
        HARNESS = "agent"
    lab = hm.new_run(task, rep, agent, review_agent, agents_file, runs_dir, limits, lead=True, harness="agent",
                     lab_class=Bootstrap)
    lab.state["lead_agent"] = lead_agent
    lab.save()
    return AgentLab(lab.dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
