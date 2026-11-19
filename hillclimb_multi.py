#!/usr/bin/env python3
"""Parallel hill-climbing harness (v2): Python is the lab manager, several workers at once.

    python hillclimb_multi.py run    tasks/<name> --agent claude [--workers 3]
    python hillclimb_multi.py resume [run]
    python hillclimb_multi.py status [run]
    python hillclimb_multi.py report [run]

Each worker is a separate agent session in its own git worktree, carrying one experiment
from plan to analysis. One measurement thread times every candidate, one at a time, so
parallel work never disturbs a benchmark. Candidates are compared with the current champion
in alternating pairs that stop early once the answer is clear; a win over an existing
champion is confirmed with fresh pairs. Task loading, environments, evaluation and the final test come
from hillclimb.py. See README.md.
"""
import argparse
import fnmatch
import heapq
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import statistics
import string
import subprocess
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
from hillclimb import AgentDown, ContractError, Fail, git, log, num, read_json, tail, write_json

HARNESS = hc.HARNESS
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
PRIORITY = {"val": 0, "check": 1, "train": 2}
ELITES, DIVERSITY_SLOTS, CHAMPION_PARENT_P = 4, 2, 0.7
VALUED = ("keep", "keep-unconfirmed", "discard")          # decisions that leave a valid candidate


class Paused(Exception):
    """The run is pausing; abandon the current experiment without recording it."""


# ---------------------------------------------------------------- prompts (v1's, adapted)

RULES = """You are one worker in an automated research lab that improves a piece of software.
$workers workers run at the same time, each in its own copy of the code. A controller script
assigns work, measures every candidate, and decides what is kept. Nobody is watching, so
never ask questions and never wait for confirmation: do the step and exit.

Rules for every step:

- Work only inside the current directory (your workspace). Do not read, list or search
  local files outside it, except paths TASK.md explicitly names as readable. Never inspect
  credentials or environment secrets. Public web research is allowed only when stated below.
- `TASK.md` describes the task. Read it first if you have not seen it in this session.
- You may edit only files matching: $editable (plus anything in `research/`). Changes to
  other files are reverted automatically. Never change `TASK.md`, `dev_eval` or `data/`.
- `data/train/` is the only data you may see. The controller scores candidates on a hidden
  validation split and you receive only the score and aggregate metrics. Do not try to
  find, guess or reconstruct the hidden data, and do not special-case the benchmark.
- `./dev_eval` builds and scores your workspace on the train split. It goes through the
  lab's single measurement queue, so it can wait while other candidates are measured. Use
  it when a measurement answers a question; never call it in a loop.
- The machine is shared with the other workers and the measurement queue. Avoid long
  benchmarks of your own: they are noisy here and slow everyone else's measurements.
- Use only the libraries the task allows. The controller evaluates in its own environment,
  so anything you install yourself will be missing there.
- Never weaken, delete, skip, or special-case correctness checks to improve a score.
  Preserve existing test coverage; add tests for new behavior and edge cases.
- Do not commit, branch or tag: the controller commits your workspace after each step.
- `research/HISTORY.md` and `research/LESSONS.md` are the lab notebook: every finished
  experiment of every worker. They are written by the controller. Read them; do not edit them.

Research principles:

- Reason from required behavior and measured costs. Separate explicit task constraints
  from assumptions inherited from the implementation. Question those assumptions while
  preserving correctness, the interface, data boundaries, and allowed dependencies.
- Prefer eliminating unnecessary work, allocations, state, and abstractions before
  making them faster. Use established techniques and permitted libraries when justified;
  understand their tradeoffs rather than reproducing a design blindly.
- Distinguish observations, estimates, and hypotheses. Never invent measurements. Report
  the command, input scope, repetitions, and observed spread for performance evidence.
- Each step has its own file-writing contract below. Writing a plan, review, or analysis
  does not authorize implementing changes during that step. Exit after writing its output.

$goal"""

PLAN_TEMPLATE = """Write the plan to `research/PLAN.md` with exactly these sections. The title starts
with a short lowercase area tag in brackets, naming the part of the system you change
(for example `[search-loop]`, `[index-build]`, `[memory-layout]`):

```
# Experiment $exp: [<area>] <short title>
## Hypothesis
## Evidence
## Proposed change
## Implementation steps
## Expected result
## Risks
## How this tests the hypothesis
```

Only write the plan in this step. Do not change any code."""

PROPOSE_FIRST = """This is experiment $exp. No experiment has finished yet; other workers may be starting
their first experiments at the same time.

$assignment

1. Read `TASK.md` and all existing code. Look at the train data in `data/train/`.
2. Run `./dev_eval` once to see how the starting code does.
3. Work out where the main cost or weakness is. If web search is available, look up known
   approaches to this kind of problem.
4. Choose the single most promising step in an area no other worker is on. If there is no
   working solution yet, that step is a simple, correct solution; speed comes later.

Make the hypothesis falsifiable: identify the bottleneck, explain how the change reduces
its cost, estimate the effect and uncertainty, and state what result would contradict the
hypothesis. Label estimates separately from timings. In Risks, cover correctness edge cases
and applicability beyond the observed train inputs.

""" + PLAN_TEMPLATE

PROPOSE_NEXT = """This is experiment $exp.

$assignment

1. Read `research/HISTORY.md` and `research/LESSONS.md`: every finished experiment of every
   worker, including failures and why. A candidate's code can be read with `git show <tag>`.
2. Re-read the parts of the code that matter and measure with `./dev_eval` where useful.
3. Propose one new hypothesis. It must not repeat an experiment that already failed unless
   you can say what is different this time, and it should not duplicate the areas other
   workers are on right now. Prefer the idea with the best expected gain for its risk.
   If `research/AGENDA.md` exists (hillclimb_agent.py), it is the research lead's current
   ranking of directions; follow it unless your evidence says otherwise.

Connect the bottleneck to a specific mechanism. State expected gain and uncertainty, what
evidence would contradict the hypothesis, and correctness and generalization risks. Do not
invent a marginal change merely to use the remaining budget.

If no worthwhile experiment remains, write `research/stop.json` instead of a plan:

```json
{"reason": "why further work is not worthwhile", "evidence": "measurements and history supporting this", "alternatives": "remaining approaches considered and why they are not worth the budget"}
```

A fresh reviewer must approve, and the lab stops only when workers agree. Do not write
stop.json when proposing an experiment.

""" + PLAN_TEMPLATE


REVIEW_CODE = """Review the code change for experiment $exp. You did not write it. See it with:

    git diff $base HEAD -- . ':(exclude)research'

Compare it with `research/PLAN.md` and `TASK.md`.

Part 1, code review:

- Does the change do what the plan says?
- Is there a correctness bug, including on inputs unlike the train data?
- Is there complexity that is not needed?

Part 2, audit for benchmark gaming. Flag the change if it does any of these, even when the
score would look good:

- special-cases the benchmark: detects train inputs, sizes, shapes or values, or hard-codes
  answers or constants that only fit the train data without a general fallback
- caches or reuses results across calls or runs, or skips work the task requires
- interferes with timing, the driver, the evaluator or the correctness checks
- lowers precision or accuracy below what TASK.md allows, or uses a library or source
  TASK.md forbids
- reads files it may not read (hidden data, other workspaces)

Write only `research/review.json`, valid JSON, and change nothing else:

```
{"verdict": "approve" | "revise", "issues": ["file and specific problem", "..."],
 "audit": "clean" | "flagged", "audit_issues": ["specific gaming concern", "..."]}
```

A revise verdict needs at least one issue; a flagged audit needs at least one audit issue.
Check that correctness tests were preserved. A requested fix receives one further independent
review; do not request cosmetic changes as correctness blockers."""

def load_prompts():
    prompts = hc.load_prompts()
    prompts.update(rules=RULES, propose_first=PROPOSE_FIRST, propose_next=PROPOSE_NEXT,
                   review_code=REVIEW_CODE)
    return prompts


# ---------------------------------------------------------------- measurement statistics

def sequential_compare(measure_pair, direction, threshold, min_pairs, max_pairs):
    """Compare a candidate with the champion in alternating pairs, stopping once the answer is clear.

    measure_pair(i) returns (champion score or None, candidate score or None) for pair i;
    None means invalid. Gains are log ratios, so a 30% speed-up and its inverse are symmetric.
    Stops early when the 95% confidence interval of the mean log gain lies entirely above
    log(1 + threshold) (keep) or entirely below it (discard). Otherwise, after max_pairs,
    decides as v1 does: median-based gain above the threshold.
    """
    sign = 1 if direction == "maximize" else -1
    bound = math.log1p(threshold)
    best, cand, logs = [], [], []
    out = {"decision": "discard", "pairs": 0, "early": False, "ci": None, "best_invalid": False}
    for i in range(max_pairs):
        b, c = measure_pair(i)
        if c is None:
            return {**out, "decision": "invalid", "pairs": i + 1, "scores": cand, "best_scores": best}
        cand.append(c)
        if b is None:
            out["best_invalid"] = True
            break
        best.append(b)
        logs.append(sign * math.log(c / b) if b > 0 and c > 0 else 0.0)
        n = len(logs)
        if n >= min(min_pairs, max_pairs) and n >= 2:
            mean, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T95.get(n - 1, 1.96) * sd / math.sqrt(n)
            out["ci"] = [math.expm1(mean - half), math.expm1(mean + half)]
            if mean - half > bound or mean + half < bound:
                out.update(early=n < max_pairs, decision="keep" if mean - half > bound else "discard")
                break
    out["pairs"] = len(cand)
    out["scores"], out["best_scores"] = cand, best
    out["score"] = statistics.median(cand)
    out["best_score"] = statistics.median(best) if best else None
    if out["best_score"]:
        g = (out["score"] - out["best_score"]) / abs(out["best_score"])
        out["gain"] = g if direction == "maximize" else -g
        if not out["early"]:
            out["decision"] = "keep" if out["gain"] > threshold else "discard"
    else:
        out["gain"] = None
    return out


def choose_parent(rng, champion, elites):
    """Champion with probability CHAMPION_PARENT_P, otherwise a random elite."""
    if elites and (champion is None or rng.random() >= CHAMPION_PARENT_P):
        return rng.choice(elites)
    return champion


# ---------------------------------------------------------------- the lab

class Request:
    _seq = itertools.count()

    def __init__(self, kind, **fields):
        self.kind, self.fields, self.seq = kind, fields, next(Request._seq)
        self.done, self.result = threading.Event(), None

    def __lt__(self, other):
        return (PRIORITY[self.kind], self.seq) < (PRIORITY[other.kind], other.seq)


class Lab:
    HARNESS = "multi"
    auto_promote = True                                   # v3 (hillclimb_agent) promotes by request instead

    def __init__(self, run_dir):
        self.dir = Path(run_dir).resolve()
        self.repo = self.dir / "repo"
        self.state = read_json(self.dir / "state.json")
        if not self.state or self.state.get("harness") != self.HARNESS:
            raise Fail(f"{self.dir} is not a hillclimb_{self.HARNESS} run directory")
        self.task = hc.load_task(self.state["task_dir"])
        self.limits = {**self.task["budget"], **self.state["limits"]}
        agents = tomllib.loads(Path(self.state["agents_file"]).read_text())
        for role in ("agent", "review_agent"):
            if self.state[role] not in agents:
                raise Fail(f"agent '{self.state[role]}' is not defined in {self.state['agents_file']}")
        self.worker_cfg, self.review_cfg = agents[self.state["agent"]], agents[self.state["review_agent"]]
        self.prompts = load_prompts()
        self.records = [json.loads(x) for x in self.records_file.read_text().splitlines() if x.strip()] \
            if self.records_file.exists() else []
        self.notes = {}
        if self.notebook_file.exists():
            for line in self.notebook_file.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.notes[entry["exp"]] = entry["analysis"]
        self.lock, self.git_lock, self.journal_lock = threading.RLock(), threading.Lock(), threading.Lock()
        self.queue, self.queue_cv = [], threading.Condition()
        self.abort, self.measure_stop = threading.Event(), threading.Event()
        self.procs, self.pause_reason, self.fatal = set(), None, None
        self.busy = {}                                    # slot -> area of its running experiment
        self.propose_lock = threading.Lock()             # proposals one at a time: each sees the others'
        self.built = {}                                   # measurement dir -> sha built there
        self.calls = itertools.count(len(list((self.dir / "logs").glob("*.log"))) + 1)
        self.rng = random.Random(f"{self.state['run_id']}-{len(self.records)}")
        self.evaluate = hc.evaluate                       # replaceable in tests
        self.t0, self.elapsed0 = time.time(), self.state.get("elapsed_s", 0)
        # Experiments claimed but never recorded (the run stopped mid-way) are redone first.
        done = {r["exp"] for r in self.records}
        self.pending = sorted(set(self.state.get("claimed", [])) - done)

    records_file = property(lambda self: self.dir / "experiments.jsonl")
    notebook_file = property(lambda self: self.dir / "notebook.jsonl")
    workers = property(lambda self: self.limits["workers"])

    def elapsed(self):
        return self.elapsed0 + time.time() - self.t0

    def save(self):
        with self.lock:
            self.state["elapsed_s"] = round(self.elapsed())
            write_json(self.dir / "state.json", self.state)

    def journal(self, event, **fields):
        line = json.dumps({"t": time.time(), "event": event, **fields})
        with self.journal_lock, open(self.dir / "journal.jsonl", "a") as f:
            f.write(line + "\n")

    def g(self, cwd, *args):
        with self.git_lock:
            return git(cwd, *args)

    # ---- population

    @property
    def champion(self):
        return self.state.get("champion")

    def elites(self):
        """Valid non-champion candidates worth branching from: the best few, plus area diversity."""
        champ = (self.champion or {}).get("sha")
        maximize = self.task["direction"] == "maximize"
        pool = [r for r in self.records if r["decision"] in VALUED and r.get("sha") and r["sha"] != champ
                and (r.get("val") or {}).get("score") is not None]
        pool.sort(key=lambda r: r["val"]["score"], reverse=maximize)
        chosen = pool[:ELITES]
        for r in pool[ELITES:]:
            if len(chosen) >= ELITES + DIVERSITY_SLOTS:
                break
            if r.get("area") not in {c.get("area") for c in chosen}:
                chosen.append(r)
        return [{"sha": r["sha"], "score": r["val"]["score"], "exp": r["exp"], "area": r.get("area")}
                for r in chosen]

    def champion_history(self):
        """Records that became champion, in order. In v2 a confirmed keep is promoted at once."""
        return [r for r in self.records if r["decision"] == "keep"]

    # ---- claiming work

    def claim(self):
        """Next experiment number for a worker, or None when no new experiment may start."""
        with self.lock:
            if self.abort.is_set() or self.stopping():
                return None
            if self.pending:
                return self.pending.pop(0)
            claimed = self.state.setdefault("claimed", [])
            if len(claimed) >= self.limits["max_experiments"] or self.elapsed() >= self.limits["max_hours"] * 3600:
                return None
            n = self.state["next_exp"]
            self.state["next_exp"] += 1
            claimed.append(n)
            self.save()
            return n

    def stopping(self):
        votes = {v["slot"] for v in self.state.get("stop_votes", [])}
        return len(votes) >= min(2, self.workers)

    def pause(self, reason):
        with self.lock:
            if not self.pause_reason:
                self.pause_reason = reason
        self.abort.set()
        for p in list(self.procs):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # ---- records

    def record(self, rec, champion=None):
        """Append a finished experiment; optionally promote its candidate. Atomic under the lock."""
        with self.lock:
            with open(self.records_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.records.append(rec)
            if champion:
                self.state["champion"] = champion
                self.g(self.repo, "branch", "-f", "best", champion["sha"])
            if rec["decision"] == "stop":
                self.state.setdefault("stop_votes", []).append({"exp": rec["exp"], "slot": rec["slot"]})
            self.save()
        self.journal("recorded", exp=rec["exp"], decision=rec["decision"], champion=bool(champion))

    def add_note(self, exp, analysis):
        with self.lock:
            with open(self.notebook_file, "a") as f:
                f.write(json.dumps({"exp": exp, "analysis": analysis}) + "\n")
            self.notes[exp] = analysis

    def history_view(self):
        """Records in v1's history format (for v1's notebook writer and prompts)."""
        with self.lock:
            out = []
            for r in self.records:
                v = r.get("val") or {}
                out.append({**r, "analysis": self.notes.get(r["exp"], {}),
                            "title": f"{r['title']} (worker {r['slot']}, parent {r.get('parent_label', '?')})",
                            "val": {"score": v.get("score"), "gain": v.get("gain")}})
            return out

    # ---- agents

    def settings_for(self, slot):
        """Claude Code settings that hide the hidden splits and other workers from this worker (M0)."""
        t = self.task["dir"]
        hidden = [t / "data" / s for s in ("val", "test", "guard")] + [t / "reference", t / "evaluator",
                  t / "check_report.json", t / "preparation.json"]
        hidden += [self.dir / n for n in ("state.json", "journal.jsonl", "experiments.jsonl",
                                          "notebook.jsonl", "experiments", "logs", "settings")]
        hidden += [self.dir / "worktrees" / f"w{k}" for k in range(1, self.workers + 1) if k != slot]
        hidden += [self.dir / "worktrees" / n for n in ("m-best", "m-cand", "lead") if f"w{slot}" != n]
        hidden = [p for p in hidden if p.exists()]
        rules = [f"{tool}(/{p}{'/**' if p.is_dir() else ''})" for p in hidden for tool in ("Read", "Edit")]
        # The file tools are not covered by the sandbox: also forbid writing to the harness itself
        # (its code, docs, tests and tasks). Run folders stay writable for the worker's own worktree.
        runs = [self.dir.parent.resolve(), hc.HOME.resolve()]
        protected = [q for q in HARNESS.iterdir() if q.name not in (".git", ".hillclimb")
                     and not any(r == q.resolve() or q.resolve() in r.parents for r in runs)]
        protected.append(HARNESS / "research")
        rules += [f"Edit(/{q}{'/**' if q.is_dir() or q.name == 'research' else ''})" for q in protected]
        # Measurement-queue waits (./dev_eval, ./lab wait) can take minutes: allow long shell commands.
        return {"env": {"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "1200000"},
                "permissions": {"deny": rules},
                "sandbox": {"enabled": True, "allowUnsandboxedCommands": False, "failIfUnavailable": True,
                            "filesystem": {"denyRead": [str(p) for p in hidden],
                                           "allowWrite": [str(self.dir / "queue" / "requests"),
                                                          str(self.dir / "lab" / "requests")]}}}

    def call_agent(self, slot, exp, step, cwd, values, cfg, session=None, feedback="", fresh=False,
                   raw_text=None, timeout=None):
        """One agent step. Returns (finished, session id). Raises Paused or AgentDown."""
        if self.abort.is_set():
            raise Paused()
        values = {"exp": exp, "editable": ", ".join(self.task["workspace"]["editable"]),
                  "workers": self.workers, **values}
        text = raw_text or (f"[hillclimb step={step} exp={exp}]\n\n"
                            + string.Template(self.prompts["rules"]).substitute(values) + "\n\n"
                            + string.Template(self.prompts[step]).substitute(values))
        if feedback:
            text += "\n\nController output validation failed. Correct the required artifact:\n" + feedback
        settings = None
        if cfg.get("settings_arg"):
            settings = self.dir / "settings" / f"w{slot}.json"
            settings.parent.mkdir(exist_ok=True)
            write_json(settings, self.settings_for(slot))

        def command(sess):
            cmd = list(cfg["command"])
            if not self.task["agent"]["web_search"]:
                cmd += cfg.get("web_search_off", [])
            if sess and cfg.get("resume"):
                cmd += cfg["resume"]
            if settings:
                cmd += cfg["settings_arg"]
            fill = {"prompt": text, "harness": str(HARNESS), "python": sys.executable,
                    "session": sess or "", "settings": str(settings or "")}
            return [re.sub(r"\{(prompt|harness|python|session|settings)\}", lambda m: fill[m.group(1)], a)
                    for a in cmd]

        logfile = self.dir / "logs" / f"{exp:04d}-w{slot}-{next(self.calls):03d}-{step}.log"
        t0 = time.time()
        sess = None if fresh else session
        for _ in (1, 2):
            rc, out, err = self.run_tracked(command(sess), cwd, hc.env_for(self.task, "dev"),
                                            timeout or self.limits["agent_step_timeout_s"])
            if self.abort.is_set():
                raise Paused()
            if rc in (0, 124) or not sess:
                break
            sess = None
        logfile.write_text(hc.redact(f"PROMPT\n{text}\n\nEXIT {rc}\n\nSTDOUT\n{out}\n\nSTDERR\n{err}\n"))
        reply = hc.read_json_text(out)
        self.journal("agent", slot=slot, exp=exp, step=step, start=t0, end=time.time(), rc=rc)
        with self.lock:
            if isinstance(reply.get(cfg.get("cost_key")), (int, float)):
                self.state["cost_usd"] = self.state.get("cost_usd", 0) + reply[cfg["cost_key"]]
        if rc not in (0, 124) or reply.get("is_error") is True:
            message = reply.get("result") if isinstance(reply.get("result"), str) else tail(out + err, 10, 800)
            cause = "the agent's usage limit was reached" if hc.usage_limited(message) else "the agent command failed"
            raise AgentDown(hc.redact(f"{cause} in step '{step}' of experiment {exp} (worker {slot}, "
                                      f"exit code {rc}):\n{message}\nTranscript: {logfile}"))
        new_session = reply.get(cfg.get("session_key")) if cfg.get("session_key") else None
        log(f"  [w{slot} exp {exp}] {step}: {'ok' if rc == 0 else 'timed out'} ({time.time() - t0:.0f}s)")
        return rc == 0, (session if fresh else (new_session or sess))

    def run_tracked(self, cmd, cwd, env, timeout):
        """Like hillclimb.sh, but the process can be killed when the lab pauses."""
        try:
            p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, errors="replace", stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        except FileNotFoundError:
            raise Fail(f"agent command not found: {cmd[0]}")
        self.procs.add(p)
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, out, err or ""
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            return 124, out + f"\n[timed out after {timeout}s]", err or ""
        finally:
            self.procs.discard(p)

    # ---- the measurement queue

    def submit(self, kind, **fields):
        """Queue a measurement and wait for its result. Raises Paused if the lab pauses first."""
        req = Request(kind, **fields)
        with self.queue_cv:
            heapq.heappush(self.queue, req)
            self.queue_cv.notify()
        while not req.done.wait(0.5):
            if self.abort.is_set() and self.measure_stop.is_set():
                raise Paused()
        if req.result is None:
            raise Paused()
        return req.result

    def measurement_loop(self):
        """The only place where anything is timed. Runs until told to stop and the queue is empty."""
        requests_dir = self.dir / "queue" / "requests"
        while True:
            for old in (self.dir / "queue" / "results").glob("*.json"):   # replies read long ago
                if time.time() - old.stat().st_mtime > 600:
                    old.unlink(missing_ok=True)
            req = None
            with self.queue_cv:
                if self.queue:
                    req = heapq.heappop(self.queue)
            if req is None:
                req = self.file_request(requests_dir)
            if req is None:
                if self.measure_stop.is_set():
                    return
                with self.queue_cv:
                    self.queue_cv.wait(0.2)
                continue
            if self.abort.is_set() and req.kind != "train":
                req.done.set()                            # result None: the waiting worker pauses
                continue
            t0 = time.time()
            try:
                req.result = getattr(self, f"measure_{req.kind}")(**req.fields)
            except Fail as e:
                req.result = {"valid": False, "error": str(e)} if req.kind == "train" else None
                if req.kind != "train":
                    self.pause(f"measurement failed: {e}")
            except Exception:
                self.fatal = traceback.format_exc()
                self.pause("internal error in the measurement thread")
            self.journal("measure", kind=req.kind, start=t0, end=time.time(), **{
                k: v for k, v in req.fields.items() if k in ("exp", "sha", "slot")})
            if req.fields.get("reply_file"):
                tmp = Path(req.fields["reply_file"]).with_suffix(".tmp")
                tmp.write_text(json.dumps(req.result or {"valid": False, "error": "lab paused"}))
                tmp.replace(req.fields["reply_file"])
            req.done.set()

    def file_request(self, requests_dir):
        """A `./dev_eval` request written by a worker's shell (it cannot reach the lab otherwise)."""
        for f in sorted(requests_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            data = read_json(f)
            f.unlink(missing_ok=True)
            reply = self.dir / "queue" / "results" / f.name
            directory = Path(str((data or {}).get("dir", ""))).resolve()
            allowed = [(self.dir / "worktrees" / f"w{k}").resolve() for k in range(1, self.workers + 1)]
            if directory not in allowed:
                reply.write_text(json.dumps({"valid": False, "error": "dev_eval must run in a worker workspace"}))
                continue
            return Request("train", directory=str(directory), reply_file=str(reply))
        return None

    def measure_train(self, directory, reply_file=None):
        ok, out = hc.build(self.task, Path(directory))
        if not ok:
            return {"valid": False, "error": "BUILD FAILED\n" + tail(out, 40, 3000)}
        r = self.evaluate(self.task, Path(directory), self.task["p"]["train"])
        return {k: r.get(k) for k in ("valid", "score", "metrics", "error")}

    def measure_check(self, directory, exp=None, slot=None):
        """Controller checks before val: build, the task's test command, the evaluator on train."""
        task, d = self.task, Path(directory)
        ok, out = hc.build(task, d)
        if not ok:
            return {"ok": False, "log": "BUILD FAILED\n" + tail(out, 60, 4000)}
        if task["commands"].get("test"):
            rc, out, _ = hc.sh(hc.task_command(task, "test"), cwd=d, env=hc.env_for(task, "eval"),
                               timeout=task["eval"]["timeout_s"])
            if rc:
                return {"ok": False, "log": f"TEST COMMAND FAILED (exit code {rc})\n" + tail(out, 60, 4000)}
        r = self.evaluate(task, d, task["p"]["train"])
        if not r["valid"]:
            return {"ok": False, "log": (f"THE EVALUATOR REJECTED THE CANDIDATE ON THE TRAIN SPLIT\n"
                                         f"error: {r['error']}\nmetrics: {json.dumps(r['metrics'])}")}
        return {"ok": True, "log": f"passed; train score {num(r['score'])}, metrics {json.dumps(r['metrics'])}"}

    def checkout_built(self, name, sha):
        d = self.dir / "worktrees" / name
        if self.built.get(name) != sha:
            with self.git_lock:
                hc.checkout_clean(d, sha)
            ok, out = hc.build(self.task, d)
            if not ok:
                self.built.pop(name, None)
                return None, out
            self.built[name] = sha
        return d, ""

    def compare(self, sha, champion):
        """Sequential paired comparison of sha with a champion (dict or None) on val."""
        task, val = self.task, self.task["p"]["val"]
        reps = task["eval"]["repetitions"]
        cand, out = self.checkout_built("m-cand", sha)
        if cand is None:
            return {"decision": "invalid", "valid": False, "error": "build failed in a clean checkout:\n"
                    + tail(out, 20, 1500), "pairs": 0}
        metrics, errors = {}, []

        def one(d):
            e = self.evaluate(task, d, val)
            if not e["valid"]:
                errors.append(e["error"] or "the evaluator reported valid=false")
                return None, e
            return e["score"], e

        if champion is None:                              # no valid solution yet: score it alone
            scores = []
            for _ in range(min(2, reps)):
                s, e = one(cand)
                metrics = e["metrics"]
                if s is None:
                    return {"decision": "invalid", "valid": False, "error": errors[-1], "pairs": len(scores) + 1,
                            "metrics": metrics}
                scores.append(s)
            return {"decision": "keep", "valid": True, "score": statistics.median(scores), "scores": scores,
                    "best_score": None, "gain": None, "pairs": len(scores), "metrics": metrics, "vs": None}
        best, out = self.checkout_built("m-best", champion["sha"])

        def pair(i):
            nonlocal metrics
            order = [("b", best), ("c", cand)] if i % 2 == 0 else [("c", cand), ("b", best)]
            got = {}
            for who, d in order:
                if who == "b" and d is None:
                    got["b"] = None
                    continue
                s, e = one(d)
                got[who] = s
                if who == "c":
                    metrics = e["metrics"]
                    if s is None:
                        break
            return got.get("b"), got.get("c")

        res = sequential_compare(pair, task["direction"], self.state["check"]["threshold"],
                                 min(2, reps), reps)
        res.update(valid=res["decision"] != "invalid", metrics=metrics, vs=champion["sha"],
                   error=errors[-1] if res["decision"] == "invalid" else None)
        if res.get("best_invalid") and res["valid"]:
            ref = champion.get("score")
            res["best_score"] = ref
            res["gain"] = hc.gain(task, res["score"], ref) if ref else None
            res["decision"] = "keep" if res["gain"] is not None and res["gain"] > self.state["check"]["threshold"] \
                else "discard"
        return res

    def measure_val(self, sha, rec, exp=None, slot=None):
        """Measure a candidate against the current champion, confirm a win, record the experiment."""
        with self.lock:
            champion = self.champion
        log(f"  [measure] exp {rec['exp']} vs {'champion exp ' + str(champion['exp']) if champion else 'nothing'}")
        v = self.compare(sha, champion)
        rec["val"] = {k: v.get(k) for k in ("valid", "score", "scores", "best_score", "best_scores", "gain",
                                            "pairs", "early", "ci", "metrics", "error", "vs")}
        rec["vs_exp"] = champion["exp"] if champion else None
        new_champion = None
        if v["decision"] == "keep" and champion is not None:
            c = self.compare(sha, champion)                 # confirmation with fresh pairs
            rec["confirm"] = {k: c.get(k) for k in ("decision", "score", "best_score", "gain", "pairs", "ci")}
            if c["decision"] == "invalid":                  # valid once, invalid now: not a usable candidate
                v["decision"] = "invalid"
                v["error"] = c.get("error") or "the candidate was not valid in the confirmation measurement"
                rec["val"].update(valid=False, error=v["error"])
            elif c["decision"] != "keep":
                v["decision"] = "keep-unconfirmed"
        threshold = self.state["check"]["threshold"]
        if v["decision"] in VALUED and self.task["p"]["guard"]:
            g = self.evaluate(self.task, self.dir / "worktrees" / "m-cand", self.task["p"]["guard"])
            rec["val"]["guard_valid"] = g["valid"]
            if not g["valid"]:
                v["decision"] = "guard-failed"
        rec["eligible"] = v["decision"] == "keep"
        if v["decision"] == "keep":
            if self.auto_promote:
                new_champion = {"sha": sha, "score": rec["val"]["score"], "exp": rec["exp"]}
            rec["reason"] = ("first valid solution" if champion is None else
                             f"gain {v['gain']:+.1%} is above the threshold {threshold:.1%} (confirmed)")
            if not self.auto_promote:
                rec["reason"] += "; eligible for promotion"
        elif v["decision"] == "keep-unconfirmed":
            cg = rec["confirm"].get("gain")
            rec["reason"] = (f"gain {v['gain']:+.1%} was above the threshold but the confirmation measured "
                             f"{'n/a' if cg is None else format(cg, '+.1%')}; kept in the population, not as champion")
        elif v["decision"] == "guard-failed":
            rec["reason"] = hc.GUARD_FAILED
        elif v["decision"] == "invalid":
            rec["reason"] = f"not valid on val: {v.get('error')}"
        else:
            rec["reason"] = (f"gain {v['gain']:+.1%} is not above the threshold {threshold:.1%}"
                             if v.get("gain") is not None else "not better than the champion")
        rec["decision"] = v["decision"]
        rec["tag"] = f"exp-{rec['exp']:04d}-{rec['decision']}"
        self.g(self.repo, "tag", "-f", rec["tag"], sha)
        rec["seconds"] = round(time.time() - rec.pop("_t0"))
        if self.auto_promote:
            self.queue_merge(rec, champion)
        self.record(rec, new_champion)
        log(f"  [w{rec['slot']} exp {rec['exp']}] {rec['decision'].upper()}: {rec['reason'].splitlines()[0][:110]}")
        return rec

    def queue_merge(self, rec, champion):
        """A candidate that beat its own (older) parent but not the champion, in a different area,
        is worth porting onto the champion: queue a merge assignment for the next free worker."""
        v, task = rec.get("val") or {}, self.task
        if (rec["decision"] not in ("discard", "keep-unconfirmed") or rec.get("merge_of") or not champion
                or rec.get("parent") == champion["sha"] or not rec.get("parent_score") or not v.get("score")):
            return
        if hc.gain(task, v["score"], rec["parent_score"]) <= self.state["check"]["threshold"]:
            return
        with self.lock:
            champ_area = next((r.get("area") for r in self.records if r.get("sha") == champion["sha"]), None)
            if champ_area == rec.get("area"):
                return
            self.state.setdefault("merges", []).append({k: rec.get(k) for k in ("exp", "tag", "parent", "area",
                                                                                "title")})
        log(f"  [lab] merge queued: experiment {rec['exp']} beat its parent; port it onto the champion")

    # ---- one experiment

    def prepare(self, wt, sha):
        with self.git_lock:
            git(wt, "checkout", "-q", "-f", "--detach", sha)
            git(wt, "clean", "-fdq")                      # keeps ignored files: data/, research/
        research = wt / "research"
        shutil.rmtree(research, ignore_errors=True)
        research.mkdir()
        hc.write_history(SimpleNamespace(history=self.history_view(), ws=wt))
        if (self.dir / "agenda.md").exists():
            shutil.copy(self.dir / "agenda.md", research / "AGENDA.md")

    def commit(self, wt, base):
        """Commit the worker's changes and revert anything outside the editable set."""
        with self.git_lock:
            hc.snapshot(wt)
            changed = git(wt, "diff", "--name-only", "--no-renames", base, "HEAD").splitlines()
            allowed = self.task["workspace"]["editable"] + hc.ALWAYS_EDITABLE
            bad = [f for f in changed if any(fnmatch.fnmatch(f, p) for p in hc.PROTECTED)
                   or not any(fnmatch.fnmatch(f, p) for p in allowed)]
            for f in bad:
                if hc.git_ok(wt, "cat-file", "-e", f"{base}:{f}"):
                    git(wt, "checkout", base, "--", f)
                else:
                    git(wt, "rm", "-q", "-f", "--", f)
            if bad:
                hc.snapshot(wt, "revert files outside the editable set")

    def squash(self, wt, base, msg):
        with self.git_lock:
            hc.snapshot(wt)
            if git(wt, "rev-parse", "HEAD") == base:
                return None
            git(wt, "reset", "-q", "--soft", base)
            git(wt, "commit", "-q", "--allow-empty", "-m", msg)
            return git(wt, "rev-parse", "HEAD")

    def goal(self):
        rep, task = self.state["check"], self.task
        lines = [f"The score is {'higher' if task['direction'] == 'maximize' else 'lower'}-is-better.",
                 "Public web research is " + ("enabled." if task["agent"]["web_search"] else "disabled.")]
        if rep["baseline"] == "reference":
            lines.append(f"Reference to beat (its implementation is hidden): val score {num(rep['scores'].get('val'))}.")
        c = self.champion
        if c:
            lines.append(f"Current champion: experiment {c['exp'] or 'start'}, val score {num(c['score'])}. A "
                         f"candidate replaces it only if it is valid and beats the champion at the time it is "
                         f"measured by more than {rep['threshold']:.1%} on the hidden val split, twice.")
        else:
            lines.append("There is no valid solution yet. The first valid candidate becomes the champion.")
        return "\n".join(lines)

    def experiment(self, slot, n, parent=None, idea=None, merge=None):
        wt = self.dir / "worktrees" / f"w{slot}"
        xd = self.dir / "experiments" / f"{n:04d}"
        shutil.rmtree(xd, ignore_errors=True)
        xd.mkdir(parents=True)
        with self.lock:
            champion = self.champion
            if parent is None:                            # v2: Python chooses; v3 passes the lead's choice
                merge = self.state["merges"].pop(0) if champion and self.state.get("merges") else None
                parent = champion if merge else choose_parent(self.rng, champion, self.elites())
            parent = parent or {"sha": self.state["start_sha"], "exp": 0, "score": None}
            self.save()
        base = parent["sha"]
        label = "the start" if not parent.get("exp") else f"experiment {parent['exp']}"
        is_champion = champion is not None and parent["sha"] == champion["sha"]
        assignment = (f"Your starting point is {label} ({'the current champion' if is_champion else 'an elite, not the champion'}"
                      f", commit {base[:12]}, val score {num(parent.get('score'))}). Your workspace holds its code.\n"
                      "$busy")
        if merge:
            assignment = (f"Your assignment is a MERGE. Experiment {merge['exp']} ({merge['title']}) improved on "
                          f"its own starting point but was measured against an older champion. Port its change "
                          f"onto the current champion, which is in your workspace (commit {base[:12]}). See the "
                          f"change with `git diff {merge['parent'][:12]} {merge['tag']}`. Plan how to combine both "
                          f"improvements; use the area tag `[merge]`.\n$busy")
        if idea:
            assignment += f"\nThe research lead asks you to work on this: {idea}"
        self.journal("start", slot=slot, exp=n, parent=base)
        log(f"[w{slot}] experiment {n} from {label}")
        self.prepare(wt, base)
        plan, stop_file = wt / "research" / "PLAN.md", wt / "research" / "stop.json"
        rec = {"exp": n, "slot": slot, "parent": base, "parent_label": label, "parent_exp": parent.get("exp"),
               "decision": "rejected", "reason": "", "title": "", "hypothesis": "", "area": None, "val": None,
               "sha": None, "tag": None, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "_t0": time.time(),
               "parent_score": parent.get("score"), "merge_of": merge["exp"] if merge else None}
        common = {"base": base, "goal": self.goal(), "assignment": assignment}

        def busy_text():
            with self.lock:
                others = [a for s, a in sorted(self.busy.items()) if s != slot and a]
            return ("Other workers are working on these right now; do not duplicate them: "
                    + ("; ".join(others) if others else "none") + ".")
        session = None

        def work(step, feedback="", **values):
            nonlocal session
            _, session = self.call_agent(slot, n, step, wt, {**common, **values}, self.worker_cfg,
                                         session=session, feedback=feedback)
            self.commit(wt, base)

        def review(step, **values):
            f = wt / "research" / "review.json"
            head = git(wt, "rev-parse", "HEAD")
            saved = plan.read_text() if plan.exists() else None
            allowed = ("approve", "revise") if step == "review_code" else hc.VERDICTS
            feedback = ""
            for _ in range(2):
                f.unlink(missing_ok=True)
                try:
                    self.call_agent(slot, n, step, wt, {**common, **values}, self.review_cfg,
                                    feedback=feedback, fresh=True)
                    v = read_json(f)
                finally:
                    with self.git_lock:
                        git(wt, "checkout", "-q", "-f", "--detach", head)
                        git(wt, "clean", "-fdq")
                    if saved is not None:
                        plan.write_text(saved)
                if (isinstance(v, dict) and v.get("verdict") in allowed and isinstance(v.get("issues"), list)
                        and all(isinstance(i, str) and i.strip() for i in v["issues"])
                        and (v["verdict"] == "approve" or v["issues"])):
                    if step != "review_code":
                        return {"verdict": v["verdict"], "issues": v["issues"]}
                    audit_issues = v.get("audit_issues", [])
                    if (v.get("audit") in ("clean", "flagged") and isinstance(audit_issues, list)
                            and all(isinstance(i, str) and i.strip() for i in audit_issues)
                            and (v["audit"] == "clean" or audit_issues)):
                        return {"verdict": v["verdict"], "issues": v["issues"], "audit": v["audit"],
                                "audit_issues": audit_issues}
                feedback = (f"Write research/review.json with verdict in {allowed} and an issues array of "
                            "non-empty strings. A non-approve verdict requires at least one issue."
                            + (" Also give audit as clean or flagged, and an audit_issues array; flagged "
                               "requires at least one audit issue." if step == "review_code" else ""))
            raise ContractError(f"{step}: no valid review after two attempts (experiment {n}, worker {slot})")

        def plan_step(step, **values):
            feedback = ""
            for _ in range(2):
                common["assignment"] = assignment.replace("$busy", busy_text())
                work(step, feedback=feedback, **values)
                request = read_json(stop_file)
                if step.startswith("propose") and stop_file.exists():
                    if (not plan.exists() and champion is not None and isinstance(request, dict)
                            and all(isinstance(request.get(k), str) and request[k].strip()
                                    for k in ("reason", "evidence", "alternatives"))):
                        return request
                    feedback = ("Write either a plan or a convergence request, never both. A convergence "
                                "request requires a champion and non-empty reason, evidence and alternatives.")
                else:
                    feedback = hc.plan_problem(plan)
                    if feedback is None:
                        return None
            raise ContractError(f"{step}: invalid planning output after two attempts: {feedback}")

        def stages():
            """Returns None when the candidate is ready for val, else the reason it stops here."""
            with self.lock:
                first = not self.records
            with self.propose_lock:                       # the next proposer sees this plan's title
                request = plan_step("propose_first" if first else "propose_next")
                if request is None and plan.exists():
                    with self.lock:
                        self.busy[slot] = hc.plan_summary(plan.read_text())[0].split(":", 1)[-1].strip()
            stop_file.unlink(missing_ok=True)
            if request is not None:
                write_json(xd / "stop.json", request)
                v = review("review_stop", stop_request=json.dumps(request, indent=2))
                write_json(xd / "stop_review.json", v)
                rec["title"] = "No worthwhile next experiment"
                if v["verdict"] == "approve":
                    rec["decision"] = "stop"
                    return "convergence approved: " + request["reason"]
                rec["decision"] = "stop-rejected"
                return "convergence request not approved: " + "; ".join(v["issues"])
            rec["title"], rec["hypothesis"] = hc.plan_summary(plan.read_text())
            m = re.search(r"\[([a-z0-9][a-z0-9 _-]{0,40})\]", rec["title"])
            rec["area"] = m.group(1).strip() if m else "general"
            reviews = []
            for round_no in range(self.limits["max_plan_revisions"] + 1):
                v = review("review_plan")
                reviews.append(v)
                if v["verdict"] != "revise" or round_no == self.limits["max_plan_revisions"]:
                    break
                plan_step("revise_plan", issues="\n".join(f"- {i}" for i in v["issues"]))
            write_json(xd / "plan_review.json", reviews)
            shutil.copy(plan, xd / "plan.md")
            if v["verdict"] == "reject":
                return "plan rejected by review: " + "; ".join(v["issues"])[:300]
            if v["verdict"] != "approve":
                return "plan revisions exhausted without approval: " + "; ".join(v["issues"])[:300]
            work("implement")
            if not hc.code_changed(wt, base):
                return "no code change"
            v = review("review_code")
            code_reviews = [v]
            if v["verdict"] != "approve" and v["issues"]:
                work("fix_review", issues="\n".join(f"- {i}" for i in v["issues"]))
                v = review("review_code")
                code_reviews.append(v)
            write_json(xd / "code_review.json", code_reviews)
            if v.get("audit") == "flagged":
                rec["decision"] = "flagged"
                return "flagged by the audit: " + "; ".join(v["audit_issues"])[:300]
            if v["verdict"] != "approve":
                return "code review still requires changes: " + "; ".join(v["issues"])[:300]
            for attempt in range(self.limits["max_fix_attempts"] + 1):
                r = self.submit("check", directory=str(wt), exp=n, slot=slot)
                with open(xd / "test.log", "a") as f:
                    f.write(f"--- attempt {attempt + 1}: {'pass' if r['ok'] else 'fail'}\n{r['log']}\n\n")
                if r["ok"]:
                    return None
                if attempt < self.limits["max_fix_attempts"]:
                    work("fix_tests", log=r["log"])
            return f"tests still failing after {self.limits['max_fix_attempts']} fix attempt(s)"

        try:
            reason = stages()
            if plan.is_file():
                shutil.copy(plan, xd / "plan.md")
            sha = self.squash(wt, base, f"exp {n:04d}: {rec['title'] or 'untitled'}")
            rec["sha"] = sha
            if sha:
                (xd / "diff.patch").write_text(git(wt, "diff", base, sha, "--", ".", ":(exclude)research") + "\n")
            if reason is None:
                rec = self.submit("val", sha=sha, rec=rec, exp=n, slot=slot)
                write_json(xd / "eval.json", {"val": rec["val"], "confirm": rec.get("confirm")})
            else:
                rec["reason"] = reason
                if sha:
                    rec["tag"] = f"exp-{n:04d}-{rec['decision']}"
                    self.g(self.repo, "tag", "-f", rec["tag"], sha)
                rec["seconds"] = round(time.time() - rec.pop("_t0"))
                self.record(rec)
                log(f"  [w{slot} exp {n}] {rec['decision'].upper()}: {reason.splitlines()[0][:110]}")
        finally:
            with self.lock:
                self.busy.pop(slot, None)
        self.analyze(slot, n, wt, xd, rec, session, common)
        return rec["decision"]

    def analyze(self, slot, n, wt, xd, rec, session, common):
        """The worker's own session records what the experiment taught (the notebook entry)."""
        plan_text = (xd / "plan.md").read_text() if (xd / "plan.md").exists() else \
            json.dumps(read_json(xd / "stop.json", {}), indent=2)
        if not plan_text.strip() or self.abort.is_set():
            return
        v = rec.get("val")
        lines = [f"Decision: {rec['decision'].upper()}", f"Reason: {rec['reason']}"]
        if v:
            lines += [f"Valid on val: {v['valid']}", f"Candidate val score (median): {num(v.get('score'))}",
                      f"Champion val score, measured alongside: {num(v.get('best_score'))}",
                      f"Measurement pairs: {v.get('pairs')}", f"Aggregate metrics: {json.dumps(v.get('metrics'))}"]
            if v.get("gain") is not None:
                lines.append(f"Relative gain: {v['gain']:+.2%} (needed: more than {self.state['check']['threshold']:.1%})")
            if rec.get("confirm"):
                lines.append(f"Confirmation: {rec['confirm']['decision']}, gain {num(rec['confirm'].get('gain'))}")
        analysis_file = wt / "research" / "analysis.json"
        feedback = ""
        for _ in range(2):
            analysis_file.unlink(missing_ok=True)
            self.call_agent(slot, n, "analyze", wt, {**common, "result": "\n".join(lines), "plan_text": plan_text,
                                                     "tag": rec.get("tag") or "(nothing was committed)"},
                            self.worker_cfg, session=session, feedback=feedback)
            a = read_json(analysis_file)
            if (isinstance(a, dict) and a.get("supported") in ("yes", "no", "unclear")
                    and all(isinstance(a.get(k), str) and a[k].strip() for k in ("explanation", "lesson", "next"))):
                write_json(xd / "analysis.json", a)
                self.add_note(n, a)
                return
            feedback = ("Write research/analysis.json with supported equal to yes, no, or unclear, "
                        "and non-empty explanation, lesson, and next strings.")
        log(f"  [w{slot} exp {n}] no valid analysis; the notebook entry is missing")

    # ---- workers and the run

    def worker(self, slot):
        try:
            while True:
                n = self.claim()
                if n is None:
                    return
                if self.experiment(slot, n) == "stop":
                    log(f"[w{slot}] convergence approved; this worker retires")
                    return
        except Paused:
            return
        except (AgentDown, ContractError) as e:
            self.pause(str(e))
        except Exception:
            self.fatal = traceback.format_exc()
            self.pause(f"internal error in worker {slot}")


# ---------------------------------------------------------------- run lifecycle

def new_run(task, rep, agent, review_agent, agents_file, runs_dir, limits, lead=False, harness="multi",
            lab_class=None):
    run_dir = Path(runs_dir).resolve() / f"{task['name']}-{harness}-{time.strftime('%Y%m%d-%H%M%S')}"
    repo = run_dir / "repo"
    for sub in ("experiments", "logs", "worktrees", "queue/requests", "queue/results", "lab/requests",
                "lab/results"):
        (run_dir / sub).mkdir(parents=True)
    shutil.copytree(task["p"]["target"], repo)
    (repo / "TASK.md").write_text(hc.task_text(task))
    (repo / "dev_eval").write_text(
        "#!/bin/sh\n# Builds and scores this workspace on the train split, through the lab's measurement queue.\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(HARNESS / 'hillclimb_multi.py'))} "
        f"lab-train {shlex.quote(str(run_dir))} \"$(pwd)\"\n")
    (repo / "dev_eval").chmod(0o755)
    with open(repo / ".gitignore", "a") as f:
        f.write("\ndata/\nresearch/\n__pycache__/\n")
    git(repo, "init", "-q", "-b", "main")
    hc.snapshot(repo, "start: initial target")
    git(repo, "tag", "start")
    start = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "--detach")
    git(repo, "branch", "-D", "main")
    git(repo, "branch", "best", start)
    for k in range(1, limits["workers"] + 1):
        wt = run_dir / "worktrees" / f"w{k}"
        git(repo, "worktree", "add", "-q", "--detach", str(wt), start)
        shutil.copytree(task["p"]["train"], wt / "data" / "train")
    for name in ("m-best", "m-cand") + (("lead",) if lead else ()):   # lead: hillclimb_agent.py's workspace
        git(repo, "worktree", "add", "-q", "--detach", str(run_dir / "worktrees" / name), start)
    champion = {"sha": start, "score": rep["target_val"], "exp": 0} if rep["state"] == "existing" else None
    write_json(run_dir / "state.json", {
        "harness": harness, "run_id": run_dir.name, "task_dir": str(task["dir"]), "agent": agent,
        "review_agent": review_agent, "agents_file": str(agents_file), "limits": limits, "check": rep,
        "status": "running", "next_exp": 1, "claimed": [], "start_sha": start, "champion": champion,
        "stop_votes": [], "merges": [], "cost_usd": 0.0, "elapsed_s": 0,
        "final": None, "promotions": []})
    return (lab_class or Lab)(run_dir)


def start(lab):
    st = lab.state
    if st.get("final"):
        raise Fail("this run already used the test split; no more experiments are allowed")
    st["status"] = "running"
    lab.save()
    log(f"run {st['run_id']}: {lab.workers} workers, agent={st['agent']}, reviews={st['review_agent']}, "
        f"limit {lab.limits['max_experiments']} experiments or {lab.limits['max_hours']} hours")
    for kind in ("dev", "eval") + (("ref",) if lab.task["p"]["reference"] else ()):
        hc.env_for(lab.task, kind)                        # create environments before threads race for them
    meter = threading.Thread(target=lab.measurement_loop, name="measure", daemon=True)
    meter.start()
    threads = [threading.Thread(target=lab.worker, args=(k,), name=f"w{k}", daemon=True)
               for k in range(1, lab.workers + 1)]
    for t in threads:
        t.start()
        time.sleep(0.2)                                   # staggered starts: distinct proposals
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.5)
    except KeyboardInterrupt:
        lab.pause("interrupted")
        for t in threads:
            t.join()
    lab.measure_stop.set()
    meter.join()
    if lab.fatal:
        st["status"] = "paused"
        lab.save()
        raise Fail(f"internal error; run paused\n{lab.fatal}")
    if lab.pause_reason:
        st["status"] = "paused"
        lab.save()
        report(lab)
        raise Fail(f"{lab.pause_reason}\n\nRun paused; unfinished experiments are not recorded and the test "
                   f"split is untouched. Continue with: python hillclimb_multi.py resume {lab.dir}")
    st["status"] = "converged" if lab.stopping() else "finished"
    lab.save()
    if lab.task["p"]["test"]:
        final(lab)
    print()
    print(report(lab))
    print(f"report: {lab.dir / 'report.md'}")


def final(lab):
    """Evaluate the reference, the start (or first valid) and the champion on the test split, once."""
    st, task = lab.state, lab.task
    if st.get("final"):
        raise Fail("the test split was already used for this run; see report.md")
    reps = task["eval"]["repetitions"]
    keeps = [r for r in lab.records if r["decision"] == "keep"]
    try:
        m = _final_measure(lab, task, st, keeps, reps)
    finally:
        git(lab.repo, "worktree", "prune")                # the temporary worktrees are gone now
    first = keeps[0]["sha"] if keeps else None
    if st["check"]["state"] == "scratch" and first and lab.champion and first == lab.champion["sha"] and "best" in m:
        m["first_valid"] = m["best"]
    st["final"] = {n: {k: r[k] for k in ("valid", "median", "scores", "metrics", "error")} for n, r in m.items()}
    st["final"]["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    lab.save()


def _final_measure(lab, task, st, keeps, reps):
    """Build the candidates in temporary worktrees and measure them on the test split."""
    with tempfile.TemporaryDirectory() as tmp:
        cands = {}
        if task["p"]["reference"]:
            ref = Path(shutil.copytree(task["p"]["reference"], Path(tmp) / "reference"))
            if hc.build(task, ref, "ref")[0]:
                cands["reference"] = (ref, "ref")

        def built(name, sha):
            d = Path(tmp) / name
            git(lab.repo, "worktree", "add", "-q", "--detach", str(d), sha)
            return d if hc.build(task, d)[0] else None

        first = keeps[0]["sha"] if keeps else None
        if st["check"]["state"] == "existing":
            d = built("start", st["start_sha"])
            if d:
                cands["start"] = (d, "eval")
        elif first and lab.champion and first != lab.champion["sha"]:
            d = built("first_valid", first)
            if d:
                cands["first_valid"] = (d, "eval")
        if lab.champion:
            d = built("best", lab.champion["sha"])
            if d:
                cands["best"] = (d, "eval")
        log(f"final: evaluating {', '.join(cands) or 'nothing'} on the test split, {reps} runs each")
        return hc.measure(task, cands, task["p"]["test"], reps)


def report(lab):
    st, rep, task = lab.state, lab.state["check"], lab.task
    recs = lab.records
    keeps = [r for r in recs if r["decision"] == "keep"]
    champions = lab.champion_history()
    c = lab.champion
    out = [f"# Run {st['run_id']} (hillclimb_multi)", "",
           f"- Task: {task['name']} ({task['direction']})",
           f"- Workers: {lab.workers}; agent: {st['agent']}; reviews: {st['review_agent']}",
           f"- Status: {st['status']}, {len(recs)} experiment(s), {len(keeps)} new champion(s)",
           f"- Time: {st.get('elapsed_s', 0) / 3600:.2f} hours",
           f"- Agent cost reported: ${st.get('cost_usd', 0):.2f}" if lab.worker_cfg.get("cost_key")
           else "- Agent cost: not reported by this agent CLI",
           f"- Start: {rep['state']}; baseline: {rep['baseline'] or 'none'}, val score {num(rep['scores'].get('val'))}",
           f"- Keep threshold: {rep['threshold']:.1%} (noise {num(rep['noise'])}), confirmed with fresh pairs",
           f"- Champion: {('experiment ' + str(c['exp'])) if c and c['exp'] else ('start' if c else 'none')}, "
           f"val score {num(c['score']) if c else 'n/a'}",
           "", "## Experiments (in order of completion)", "",
           "| # | Worker | Parent | Area | Decision | Title | Val score | Gain | Pairs | Reason |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        v = r.get("val") or {}
        g = f"{v['gain']:+.1%}" if v.get("gain") is not None else ""
        out.append(f"| {r['exp']} | {r['slot']} | {r.get('parent_label', '')} | {r.get('area') or ''} | "
                   f"{r['decision']} | {r['title'][:70]} | {num(v.get('score'))} | {g} | {v.get('pairs', '')} | "
                   f"{r['reason'].splitlines()[0][:90] if r['reason'] else ''} |")
    out += ["", "## Champion over time", ""]
    out += [f"- experiment {r['exp']} (worker {r['slot']}): {num(r['val']['score'])}" for r in champions] or \
        ["- no new champion"]
    fin = st.get("final")
    if fin:
        out += ["", "## Test split (run once)", "", "| Candidate | Valid | Test score (median) |", "|---|---|---|"]
        for name in ("reference", "start", "first_valid", "best"):
            if name in fin:
                out.append(f"| {name} | {fin[name]['valid']} | {num(fin[name]['median'])} |")
        base = "reference" if "reference" in fin else "start"
        if base in fin and not fin[base]["valid"]:
            out += ["", f"The {base} is not valid on test ({fin[base]['error']}), so no speed ratio against it "
                    "is given. A valid best beats it on correctness."]
        if all(k in fin and fin[k].get("median") for k in (base, "best")):
            ratio = fin["best"]["median"] / fin[base]["median"]
            out += ["", f"Best over {base} on test: {ratio if task['direction'] == 'maximize' else 1 / ratio:.3g}x."]
        if all(k in fin and fin[k].get("valid") and fin[k].get("median") for k in ("first_valid", "best")):
            ratio = fin["best"]["median"] / fin["first_valid"]["median"]
            out += ["", f"Progress from the first valid solution: "
                    f"{ratio if task['direction'] == 'maximize' else 1 / ratio:.3g}x on test."]
    else:
        out += ["", "The test split has not been evaluated yet."]
    out += ["", "## Inspecting the run", "",
            f"- Repository: `{lab.repo}`; `git log best` is the champion line; `git tag` lists every experiment.",
            f"- Per-experiment files: `{lab.dir / 'experiments'}`; notebook: `{lab.notebook_file}`",
            f"- Timeline of agent calls and measurements: `{lab.dir / 'journal.jsonl'}`",
            f"- Agent transcripts: `{lab.dir / 'logs'}`"]
    text = "\n".join(out) + "\n"
    (lab.dir / "report.md").write_text(text)
    return text


def lab_train(run_dir, directory):
    """Client side of ./dev_eval: queue a train measurement and print its result."""
    q = Path(run_dir) / "queue"
    rid = uuid.uuid4().hex
    tmp = q / "requests" / f"{rid}.tmp"
    tmp.write_text(json.dumps({"dir": directory}))
    tmp.replace(q / "requests" / f"{rid}.json")
    reply = q / "results" / f"{rid}.json"
    deadline = time.time() + 7200
    while not reply.exists():
        if time.time() > deadline:
            raise Fail("no answer from the lab's measurement queue (is the run still going?)")
        time.sleep(0.3)
    r = json.loads(reply.read_text())
    print(json.dumps(r, indent=2))
    try:
        reply.unlink()                                    # the sandbox may forbid it; the lab cleans up
    except OSError:
        pass
    return 0 if r.get("valid") else 1


def find_run(arg, runs_dir):
    if arg:
        for cand in (Path(arg), Path(runs_dir) / arg):
            if (cand / "state.json").exists():
                return Lab(cand)
        raise Fail(f"no run found at '{arg}'")
    runs = sorted((d for d in Path(runs_dir).glob("*-multi-*") if (d / "state.json").exists()),
                  key=lambda d: (d / "state.json").stat().st_mtime)
    if not runs:
        raise Fail(f"no hillclimb_multi runs in {runs_dir}")
    return Lab(runs[-1])


def main(argv=None):
    hc.load_dotenv(HARNESS / ".env")
    ap = argparse.ArgumentParser(description="Parallel hill-climbing harness (v2)")
    ap.add_argument("--runs-dir", default=str(hc.HOME / "runs"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--agent", default="claude")
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
    lt = sub.add_parser("lab-train")
    lt.add_argument("run_dir")
    lt.add_argument("dir")
    a = ap.parse_args(argv)
    limits = {k: v for k, v in (("max_experiments", getattr(a, "max_experiments", None)),
                                ("max_hours", getattr(a, "max_hours", None))) if v is not None}
    if a.cmd == "lab-train":
        return lab_train(a.run_dir, a.dir)
    if a.cmd == "run":
        task = hc.load_task(a.task)
        rep = read_json(task["dir"] / "check_report.json")
        if task["problems"] or not rep or not rep["ok"] or rep["task_hash"] != hc.task_hash(task):
            raise Fail(f"the task has no passing, current check. Run: python hillclimb.py check {a.task}")
        agents_file = Path(a.agents_file).resolve()
        agents = tomllib.loads(agents_file.read_text())
        for name in (a.agent, a.review_agent or a.agent):
            if name not in agents:
                raise Fail(f"agent '{name}' is not defined in {agents_file}")
        if a.workers < 1:
            raise Fail("--workers must be at least 1")
        limits = {"max_experiments": task["budget"]["max_experiments"], "max_hours": task["budget"]["max_hours"],
                  **limits, "workers": a.workers}
        start(new_run(task, rep, a.agent, a.review_agent or a.agent, agents_file, a.runs_dir, limits))
        return 0
    lab = find_run(a.run, a.runs_dir)
    if a.cmd == "resume":
        lab.state["limits"].update(limits)
        lab.limits.update(limits)
        start(lab)
    elif a.cmd == "report":
        print(report(lab))
    elif a.cmd == "status":
        st, c = lab.state, lab.champion
        print(f"{st['run_id']}: {st['status']}, {len(lab.records)} experiment(s) recorded, "
              f"champion {('exp ' + str(c['exp'])) if c else 'none'} ({num(c['score']) if c else 'n/a'}), "
              f"{lab.workers} workers")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
