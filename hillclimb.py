#!/usr/bin/env python3
"""Hill-climbing harness: a coding agent improves a task, a controller measures and decides.

    python hillclimb.py check  tasks/<name>           validate a task folder
    python hillclimb.py run    tasks/<name> --agent cursor
    python hillclimb.py resume [run]                  continue an interrupted run
    python hillclimb.py status [run]
    python hillclimb.py report [run]
    python hillclimb.py final  [run]                  evaluate on the test split (once)

Sections: utilities, task, environment, evaluation, check, git, agent, history,
experiment loop, final and report, command line. See README.md.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import string
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
HOME = Path(os.environ.get("HILLCLIMB_HOME", Path.home() / ".hillclimb"))
PROTECTED = ["TASK.md", "dev_eval", "data/*"]       # the agent may never change these
ALWAYS_EDITABLE = ["research/*", ".gitignore"]
VERDICTS = ("approve", "revise", "reject")
PLAN_SECTIONS = ("Hypothesis", "Evidence", "Proposed change", "Implementation steps",
                 "Expected result", "Risks", "How this tests the hypothesis")


class Fail(Exception):
    """An error to show the user without a traceback."""


class AgentDown(Fail):
    """The agent command itself failed (usage limit, network, crash): pause, do not record."""


class ContractError(Fail):
    """An agent repeatedly failed to produce a required, valid artifact."""


# ---------------------------------------------------------------- utilities

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dotenv(path):
    """Load literal, single-line .env assignments without executing shell code.

    Existing environment variables take precedence. Values are never printed.
    """
    path = Path(path)
    if not path.exists():
        return
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, raw = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise Fail(f"invalid .env assignment at line {n}")
        try:
            parts = shlex.split(raw, comments=True)
        except ValueError:
            raise Fail(f"invalid .env quoting at line {n}") from None
        if len(parts) > 1:
            raise Fail(f"quote values containing spaces in .env at line {n}")
        os.environ.setdefault(key, parts[0] if parts else "")


def redact(text):
    for name, value in os.environ.items():
        if value and len(value) >= 8 and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD)$", name):
            text = text.replace(value, "[REDACTED]")
    return text


def sh(cmd, cwd, env=None, timeout=None, merge=True):
    """Run a command (str: via the shell, list: directly). Returns (exit code, stdout, stderr).

    The command gets its own process group so a timeout kills its children too.
    """
    p = subprocess.Popen(
        cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, text=True, errors="replace",
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge else subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err or ""
    except BaseException as e:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = p.communicate()
        if isinstance(e, subprocess.TimeoutExpired):
            return 124, out + f"\n[timed out after {timeout}s]", err or ""
        raise


def tail(text, lines=30, chars=3000):
    return "\n".join((text or "").strip().splitlines()[-lines:])[-chars:]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def num(x):
    return "n/a" if x is None else f"{x:.4g}"


def fingerprint(path):
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()


def files(d):
    return sorted(p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts)


# ---------------------------------------------------------------- task

DEFAULTS = {
    "paths": {},
    "commands": {},
    "environment": {},
    "agent": {"web_search": True},
    "workspace": {"editable": ["*"]},
    "eval": {"repetitions": 5, "min_improvement": 0.02, "timeout_s": 600},
    "budget": {"max_experiments": 30, "max_hours": 8, "max_plan_revisions": 2,
               "max_fix_attempts": 3, "agent_step_timeout_s": 1800},
}
PATHS = {"target": "target", "train": "data/train", "val": "data/val", "test": "data/test",
         "reference": "reference", "broken": "broken", "guard": "data/guard"}
OPTIONAL = ("reference", "broken", "guard")
GUARD_FAILED = ("failed the guard split: a different kind of data, checked as pass/fail only. The "
                "candidate does not generalize beyond the data it was tuned on.")


def load_task(task_dir):
    """Read task.toml. Format problems are collected in task['problems'], not raised."""
    d = Path(task_dir).resolve()
    try:
        raw = tomllib.loads((d / "task.toml").read_text())
    except OSError:
        raise Fail(f"{d / 'task.toml'} not found")
    except tomllib.TOMLDecodeError as e:
        raise Fail(f"task.toml is not valid TOML: {e}")
    problems, t = [], {"dir": d, "name": raw.get("name"), "direction": raw.get("direction")}
    for section, defaults in DEFAULTS.items():
        value = raw.get(section, {})
        if not isinstance(value, dict):
            problems.append(f"[{section}] must be a table")
            value = {}
        t[section] = {**defaults, **value}
    if not isinstance(t["name"], str) or not t["name"]:
        problems.append("name: required, a non-empty string")
        t["name"] = d.name
    if t["direction"] not in ("maximize", "minimize"):
        problems.append('direction: must be "maximize" or "minimize"')
    ev = t["commands"].get("evaluate")
    if not isinstance(ev, str) or not all(k in ev for k in ("{candidate}", "{split}", "{out}")):
        problems.append("commands.evaluate: required, and must contain {candidate}, {split} and {out}")
    for section in ("eval", "budget"):
        for key, default in DEFAULTS[section].items():
            v = t[section][key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                problems.append(f"{section}.{key}: must be a non-negative number")
                t[section][key] = default
    if t["eval"]["repetitions"] < 1:
        problems.append("eval.repetitions: must be at least 1")
    if not (isinstance(t["workspace"]["editable"], list) and t["workspace"]["editable"]):
        problems.append("workspace.editable: must be a non-empty list of patterns")
    t["p"] = {}
    for key, default in PATHS.items():
        given = key in t["paths"]
        q = (d / str(t["paths"].get(key, default))).resolve()
        if q.is_dir():
            t["p"][key] = q
        else:
            t["p"][key] = None
            if given or key not in OPTIONAL:
                problems.append(f"paths.{key}: {q} is not a directory")
    if not (d / "TASK.md").is_file():
        problems.append("TASK.md is missing")
    t["problems"] = problems
    return t


def task_text(task):
    """TASK.md as the agent sees it: {task_dir} becomes the task folder's absolute path."""
    return (task["dir"] / "TASK.md").read_text().replace("{task_dir}", str(task["dir"]))


UNHASHED = ("check_report.json", "README.md", "setup.sh")   # task files that cannot affect a measurement


def task_hash(task):
    """Fingerprint of every task file that can affect a run (TASK.md included: agents read it)."""
    h = hashlib.sha256()
    for f in files(task["dir"]):
        rel = f.relative_to(task["dir"])
        if str(rel) not in UNHASHED:
            h.update(f"{rel}:{fingerprint(f)}\n".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- environment

def env_for(task, kind):
    """Environment variables for a command. kind: dev (agent), eval (controller), ref (reference).

    With an [environment] table each kind gets its own virtual environment, so packages the
    agent installs while developing are not available when the controller evaluates.
    """
    spec, env = task["environment"], dict(os.environ)
    env["HILLCLIMB_EVAL_KIND"] = kind
    if not spec:
        return env
    packages = list(spec.get("packages", []))
    if kind == "ref":
        packages += spec.get("reference_packages", [])
    key = hashlib.sha256(json.dumps([spec.get("python"), packages, spec.get("setup")]).encode())
    venv = HOME / "envs" / f"{task['name']}-{kind}-{key.hexdigest()[:10]}"
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env['PATH']}"
    if not (venv / ".ready").exists():
        log(f"creating {kind} environment: {venv}")
        shutil.rmtree(venv, ignore_errors=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        py = ["--python", str(spec["python"])] if spec.get("python") else []
        steps = [["uv", "venv", "--seed", *py, str(venv)]]
        if packages:
            steps.append(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *packages])
        if spec.get("setup"):
            steps.append(spec["setup"])
        for step in steps:
            rc, out, _ = sh(step, cwd=task["dir"], env=env)
            if rc:
                raise Fail(f"environment setup failed: {step}\n{tail(out)}")
        (venv / ".ready").touch()
    return env


# ---------------------------------------------------------------- evaluation

def task_command(task, name):
    """A command from [commands] with {task} replaced by the task directory, or None."""
    cmd = task["commands"].get(name)
    return cmd.replace("{task}", shlex.quote(str(task["dir"]))) if cmd else None


def build(task, candidate, kind="eval"):
    cmd = task_command(task, "build")
    if not cmd:
        return True, ""
    rc, out, _ = sh(cmd, cwd=candidate, env=env_for(task, kind), timeout=task["eval"]["timeout_s"])
    return rc == 0, out


def evaluate(task, candidate, split, kind="eval"):
    """One measurement of one candidate on one split. Always returns a result dict."""
    res = {"valid": False, "score": None, "metrics": {}, "error": None}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = task["commands"]["evaluate"]
        for key, value in (("task", task["dir"]), ("candidate", candidate), ("split", split), ("out", out)):
            cmd = cmd.replace("{" + key + "}", shlex.quote(str(value)))
        t0 = time.time()
        rc, text, _ = sh(cmd, cwd=candidate, env=env_for(task, kind), timeout=task["eval"]["timeout_s"])
        res["seconds"] = round(time.time() - t0, 3)
        data = read_json(out)
    if not isinstance(data, dict):
        res["error"] = f"evaluator wrote no result JSON (exit code {rc})\n{tail(text, 15, 1500)}"
        return res
    score, metrics, error = data.get("score"), data.get("metrics", {}), data.get("error")
    if not isinstance(data.get("valid"), bool) or not isinstance(metrics, dict):
        res["error"] = "evaluator result must have a boolean 'valid' and an object 'metrics'"
    elif data["valid"] and (isinstance(score, bool) or not isinstance(score, (int, float))):
        res["error"] = "evaluator result is valid but 'score' is not a number"
    else:
        res.update(valid=data["valid"], score=score if data["valid"] else None, metrics=metrics,
                   error=None if error is None else tail(str(error), 20, 1500))
    return res


def measure(task, candidates, split, reps):
    """Measure candidates alternately. candidates: {name: (directory, env kind)}.

    A candidate that is invalid once stays invalid and is not run again. Returns per name:
    scores, median, valid, metrics, error, seconds.
    """
    out = {n: {"scores": [], "valid": True, "metrics": {}, "error": None, "seconds": []} for n in candidates}
    for _ in range(reps):
        for name, (directory, kind) in candidates.items():
            r = out[name]
            if not r["valid"]:
                continue
            e = evaluate(task, directory, split, kind)
            r["seconds"].append(e["seconds"])
            r["metrics"] = e["metrics"]
            if e["valid"]:
                r["scores"].append(e["score"])
            else:
                r["valid"], r["error"] = False, e["error"] or "the evaluator reported valid=false"
        if "cand" in out and not out["cand"]["valid"]:
            break
    for r in out.values():
        r["median"] = statistics.median(r["scores"]) if r["valid"] and r["scores"] else None
    return out


def gain(task, new, old):
    """Relative improvement of new over old; positive is better."""
    g = (new - old) / abs(old) if old else float("inf")
    return g if task["direction"] == "maximize" else -g


# ---------------------------------------------------------------- check

def check(task_dir, reps=5):
    """Validate a task folder and measure its baseline. Writes check_report.json."""
    task = load_task(task_dir)
    problems, warnings = list(task["problems"]), []
    rep = {"ok": False, "task": task["name"], "state": None, "baseline": None, "scores": {},
           "target_val": None, "noise": None, "threshold": task["eval"]["min_improvement"],
           "eval_seconds": None}
    p = task["p"]

    def step(text):
        log(f"check: {text}")

    step("configuration and paths")
    prints = {}
    for split in ("train", "val", "test", "guard"):
        if p[split]:
            prints[split] = {fingerprint(f) for f in files(p[split])}
            if not prints[split]:
                problems.append(f"{split} split is empty: {p[split]}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test"), ("train", "guard"), ("val", "guard"),
                 ("test", "guard")):
        if a in prints and b in prints:
            if p[a] == p[b]:
                problems.append(f"{a} and {b} are the same directory")
            elif prints[a] & prints[b]:
                problems.append(f"{a} and {b} share {len(prints[a] & prints[b])} identical file(s)")

    if not problems:
        with tempfile.TemporaryDirectory() as tmp:
            def copy(name):
                return Path(shutil.copytree(p[name], Path(tmp) / name))

            step("build and evaluate the target on train")
            target = copy("target")
            built, blog = build(task, target)
            r = evaluate(task, target, p["train"]) if built else {"valid": False}
            rep["state"] = "existing" if r["valid"] else "scratch"
            step(f"starting state: {rep['state']}")
            if rep["state"] == "existing":
                rep["target_val"] = evaluate(task, target, p["val"])["score"]
                if task["commands"].get("test"):
                    rc, out, _ = sh(task_command(task, "test"), cwd=target, env=env_for(task, "eval"),
                                    timeout=task["eval"]["timeout_s"])
                    if rc:
                        problems.append(f"test command fails on the target:\n{tail(out, 15)}")
            elif not built and not p["reference"]:
                warnings.append(f"build fails on the target:\n{tail(blog, 10)}")

            base = None
            if p["reference"]:
                rep["baseline"], base, kind = "reference", copy("reference"), "ref"
                ok, out = build(task, base, kind)
                if not ok:
                    problems.append(f"build fails on the reference:\n{tail(out, 15)}")
                    base = None
            elif rep["state"] == "existing":
                rep["baseline"], base, kind = "target", target, "eval"

            if base:
                step(f"evaluate the baseline ({rep['baseline']}) on train, val and test")
                for split in ("train", "val", "test"):
                    r = evaluate(task, base, p[split], kind)
                    if not r["valid"] and split == "test" and kind == "ref":
                        # A reference tuned on train may itself fail to generalise; that is
                        # a finding to report, not a broken task. Train and val must pass.
                        warnings.append(f"reference is not valid on test: {r['error']}")
                    elif not r["valid"]:
                        problems.append(f"baseline is not valid on {split}: {r['error']}")
                    rep["scores"][split] = r["score"]
                    if split == "val":
                        rep["eval_seconds"] = r["seconds"]
            elif not problems:
                warnings.append("no valid baseline (no reference, and the target is not valid): "
                                "noise is not measured and the threshold is eval.min_improvement")

            if base and not problems:
                step(f"measure noise: {reps} runs of the baseline on val")
                scores = [rep["scores"]["val"]]
                scores += [evaluate(task, base, p["val"], kind)["score"] for _ in range(reps - 1)]
                scores = [s for s in scores if s is not None]
                if len(scores) < reps:
                    problems.append("baseline is not stable: some repeated val runs were invalid")
                elif len(scores) > 1 and statistics.median(scores):
                    rep["noise"] = statistics.stdev(scores) / abs(statistics.median(scores))
                    rep["threshold"] = max(task["eval"]["min_improvement"], 2 * rep["noise"])
                    if rep["noise"] > 0.05:
                        warnings.append(f"noisy benchmark: run-to-run spread is {rep['noise']:.1%}")

            if base and p["guard"] and not problems:
                step(f"evaluate the baseline ({rep['baseline']}) on the guard split")
                r = evaluate(task, base, p["guard"], kind)
                rep["guard_baseline_valid"] = r["valid"]
                if not r["valid"]:
                    warnings.append(f"baseline is not valid on the guard split: {r['error']}")

            if p["broken"] and not problems:
                step("the evaluator must reject the broken candidate")
                broken = copy("broken")
                if build(task, broken)[0] and evaluate(task, broken, p["val"])["valid"]:
                    problems.append("the evaluator accepted the broken candidate as valid")

        if rep["eval_seconds"]:
            per_exp = rep["eval_seconds"] * (2 * task["eval"]["repetitions"] + 1)
            if per_exp > 1800:
                warnings.append(f"evaluation takes about {per_exp / 60:.0f} minutes per experiment")

    rep.update(ok=not problems, problems=problems, warnings=warnings, task_hash=task_hash(task),
               checked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(task["dir"] / "check_report.json", rep)
    for w in warnings:
        print(f"  WARNING  {w}")
    for q in problems:
        print(f"  PROBLEM  {q}")
    if rep["ok"]:
        print(f"OK  state={rep['state']}  baseline={rep['baseline']}  "
              f"val={num(rep['scores'].get('val'))}  noise={num(rep['noise'])}  "
              f"threshold={rep['threshold']:.1%}  eval={num(rep['eval_seconds'])}s")
    else:
        print(f"FAILED  {len(problems)} problem(s)")
    return rep


# ---------------------------------------------------------------- git

GIT = ["git", "-c", "user.name=hillclimb", "-c", "user.email=hillclimb@localhost",
       "-c", "commit.gpgsign=false", "-c", "advice.detachedHead=false"]


def git(cwd, *args):
    p = subprocess.run([*GIT, *args], cwd=cwd, capture_output=True, text=True, errors="replace")
    if p.returncode:
        raise Fail(f"git {' '.join(args)} failed in {cwd}:\n{p.stderr.strip()}")
    return p.stdout.strip()


def git_ok(cwd, *args):
    return subprocess.run([*GIT, *args], cwd=cwd, capture_output=True).returncode == 0


def snapshot(ws, msg="wip"):
    """Commit everything in the workspace, if anything changed."""
    git(ws, "add", "-A")
    if not git_ok(ws, "diff", "--cached", "--quiet"):
        git(ws, "commit", "-q", "-m", msg)


def restore(ws, sha="HEAD"):
    """Put the workspace on branch `best` at sha and drop every uncommitted change."""
    git(ws, "checkout", "-q", "-f", "-B", "best", sha)
    git(ws, "clean", "-fdq")


def squash(ws, base, msg):
    """Turn all work since base into one commit. Returns its sha, or None if nothing changed."""
    snapshot(ws)
    if git(ws, "rev-parse", "HEAD") == base:
        return None
    git(ws, "reset", "-q", "--soft", base)
    git(ws, "commit", "-q", "--allow-empty", "-m", msg)
    sha = git(ws, "rev-parse", "HEAD")
    restore(ws, sha)
    return sha


def checkout_clean(worktree, sha):
    git(worktree, "checkout", "-q", "-f", "--detach", sha)
    git(worktree, "clean", "-fdxq")


def code_changed(ws, base):
    return bool(git(ws, "diff", "--name-only", base, "HEAD", "--", ".", ":(exclude)research"))


def enforce_editable(run, base):
    """Revert committed changes to files the agent may not edit."""
    ws, allowed = run.ws, run.task["workspace"]["editable"] + ALWAYS_EDITABLE
    # --no-renames: a renamed protected file must show up as deleted, not only under its new name
    changed = git(ws, "diff", "--name-only", "--no-renames", base, "HEAD").splitlines()
    bad = [f for f in changed
           if any(fnmatch.fnmatch(f, pat) for pat in PROTECTED)
           or not any(fnmatch.fnmatch(f, pat) for pat in allowed)]
    for f in bad:
        if git_ok(ws, "cat-file", "-e", f"{base}:{f}"):
            git(ws, "checkout", base, "--", f)
        else:
            git(ws, "rm", "-q", "-f", "--", f)
    if bad:
        snapshot(ws, "revert files outside the editable set")
        log(f"  reverted changes outside the editable set: {', '.join(bad)}")
    return bad


# ---------------------------------------------------------------- run state

class Run:
    def __init__(self, run_dir):
        self.dir = Path(run_dir).resolve()
        self.ws = self.dir / "workspace"
        self.state = read_json(self.dir / "state.json")
        if not self.state:
            raise Fail(f"{self.dir} is not a run directory (no state.json)")
        self.task = load_task(self.state["task_dir"])
        self.budget = {**self.task["budget"], **self.state.get("budget", {})}
        agents = tomllib.loads(Path(self.state["agents_file"]).read_text())
        if self.state["agent"] not in agents:
            raise Fail(f"agent '{self.state['agent']}' is not defined in {self.state['agents_file']}")
        self.agent = agents[self.state["agent"]]
        self.prompts = load_prompts()
        self.history = [json.loads(line) for line in self.records_file.read_text().splitlines()
                        if line.strip()] if self.records_file.exists() else []
        # The journal is written atomically before state.json. Recover a completed
        # experiment if the controller died between those two writes.
        if self.history and self.history[-1]["exp"] > self.state["done"]:
            last = self.history[-1]
            self.state.update(last["state_after"])
        self.calls = len(list((self.dir / "logs").glob("*.log")))   # log numbers continue on resume
        self.t0 = time.time()
        self.elapsed0 = self.state.get("elapsed_s", 0)

    @property
    def records_file(self):
        return self.dir / "experiments.jsonl"

    def elapsed(self):
        return self.elapsed0 + time.time() - self.t0

    def save(self):
        self.state["elapsed_s"] = round(self.elapsed())
        write_json(self.dir / "state.json", self.state)


def new_run(task, report, agent, agents_file, runs_dir, budget):
    run_dir = Path(runs_dir).resolve() / f"{task['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    ws = run_dir / "workspace"
    for sub in ("experiments", "logs", "worktrees"):
        (run_dir / sub).mkdir(parents=True)
    shutil.copytree(task["p"]["target"], ws)
    shutil.copytree(task["p"]["train"], ws / "data" / "train")
    (ws / "TASK.md").write_text(task_text(task))
    (ws / "research").mkdir(exist_ok=True)
    (ws / "dev_eval").write_text(
        "#!/bin/sh\n# Runs the official evaluator on the train split.\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(HARNESS / 'hillclimb.py'))} "
        f"dev-eval {shlex.quote(str(run_dir))}\n")
    (ws / "dev_eval").chmod(0o755)
    with open(ws / ".gitignore", "a") as f:
        f.write("\ndata/\n__pycache__/\n")
    git(ws, "init", "-q", "-b", "best")
    state = {
        "run_id": run_dir.name, "task_dir": str(task["dir"]), "agent": agent,
        "agents_file": str(agents_file), "budget": budget, "check": report, "status": "running",
        "done": 0, "has_best": report["state"] == "existing", "best_score": report["target_val"],
        "session": None, "cost_usd": 0.0, "elapsed_s": 0, "final": None,
    }
    write_json(run_dir / "state.json", state)
    run = Run(run_dir)
    write_history(run)
    snapshot(ws, "start: initial target")
    git(ws, "tag", "start")
    run.state["best_sha"] = git(ws, "rev-parse", "HEAD")
    for name in ("best", "cand"):
        git(ws, "worktree", "add", "-q", "--detach", str(run_dir / "worktrees" / name), "HEAD")
    run.save()
    return run


def find_run(arg, runs_dir):
    runs_dir = Path(runs_dir)
    if arg:
        for cand in (Path(arg), runs_dir / arg):
            if (cand / "state.json").exists():
                return Run(cand)
        raise Fail(f"no run found at '{arg}'")
    runs = sorted((d for d in runs_dir.glob("*") if (d / "state.json").exists()),
                  key=lambda d: (d / "state.json").stat().st_mtime)
    if not runs:
        raise Fail(f"no runs in {runs_dir}")
    return Run(runs[-1])


# ---------------------------------------------------------------- agent

def load_prompts():
    """Split prompt sections at headings outside fenced Markdown code blocks."""
    prompts, name = {}, None
    fence = None
    for line in (HARNESS / "prompts.md").read_text().splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            ticks, rest = marker.groups()
            if fence is None:
                fence = ticks
            elif ticks[0] == fence[0] and len(ticks) >= len(fence) and not rest.strip():
                fence = None
        if fence is None and line.startswith("## "):
            name = line[3:].strip()
            prompts[name] = []
        elif name:
            prompts[name].append(line)
    return {k: "\n".join(v).strip() for k, v in prompts.items()}


def run_agent(run, exp, name, fresh=False, feedback="", **values):
    """Run one agent step in the workspace. Returns False if the step timed out.

    Raises AgentDown if the agent command fails, so the run pauses instead of recording a
    failed experiment. fresh=True starts a new session that is not remembered (reviews).
    """
    cfg, st, task = run.agent, run.state, run.task
    values = {"exp": exp, "editable": ", ".join(task["workspace"]["editable"]), **values}
    text = (f"[hillclimb step={name} exp={exp}]\n\n"
            + string.Template(run.prompts["rules"]).substitute(values) + "\n\n"
            + string.Template(run.prompts[name]).substitute(values))
    if feedback:
        text += "\n\nController output validation failed. Correct the required artifact:\n" + feedback
    def command(session):
        cmd = list(cfg["command"])
        if not task["agent"]["web_search"]:
            cmd += cfg.get("web_search_off", [])
        if session and cfg.get("resume"):
            cmd += cfg["resume"]
        fill = {"prompt": text, "harness": str(HARNESS), "python": sys.executable, "session": session or ""}
        # One pass per argument, so braces inside the prompt text are never substituted.
        return [re.sub(r"\{(prompt|harness|python|session)\}", lambda m: fill[m.group(1)], a) for a in cmd]

    session = None if fresh else st.get("session")
    run.calls += 1
    logfile = run.dir / "logs" / f"{exp:04d}-{run.calls:03d}-{name}.log"
    t0 = time.time()
    for attempt in (1, 2):
        try:
            rc, out, err = sh(command(session), cwd=run.ws, env=env_for(task, "dev"),
                              timeout=run.budget["agent_step_timeout_s"], merge=False)
        except FileNotFoundError:
            raise Fail(f"agent command not found: {cfg['command'][0]}")
        if rc in (0, 124) or not session:      # done, timed out, or nothing to fall back from
            break
        session = None          # the saved session could not be resumed: start a new one
    logfile.write_text(redact(f"PROMPT\n{text}\n\nEXIT {rc}\n\nSTDOUT\n{out}\n\nSTDERR\n{err}\n"))
    reply = read_json_text(out)
    if not fresh and cfg.get("session_key") and reply.get(cfg["session_key"]):
        st["session"] = reply[cfg["session_key"]]
    if isinstance(reply.get(cfg.get("cost_key")), (int, float)):
        st["cost_usd"] = st.get("cost_usd", 0) + reply[cfg["cost_key"]]
    log(f"  agent {name}: {'ok' if rc == 0 else f'exit {rc}'} ({time.time() - t0:.0f}s)")
    if rc not in (0, 124) or reply.get("is_error") is True:
        message = reply.get("result") if isinstance(reply.get("result"), str) else tail(out + err, 10, 800)
        cause = "the agent's usage limit was reached" if usage_limited(message) else "the agent command failed"
        raise AgentDown(redact(f"{cause} in step '{name}' of experiment {exp} "
                               f"(exit code {rc}):\n{message}\nTranscript: {logfile}"))
    return rc == 0


def usage_limited(text):
    """True when an agent's error text says a usage, rate or quota limit was hit."""
    return bool(re.search(r"usage limit|rate limit|quota|limit reached|hit your .*limit|429",
                          str(text), re.IGNORECASE))


def read_json_text(text):
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def review(run, exp, name, **values):
    """A fresh agent session writes research/review.json; everything else it changes is dropped."""
    f = run.ws / "research" / "review.json"
    head = git(run.ws, "rev-parse", "HEAD")
    feedback = ""
    allowed = ("approve", "revise") if name == "review_code" else VERDICTS
    for _ in range(2):
        f.unlink(missing_ok=True)
        try:
            run_agent(run, exp, name, fresh=True, feedback=feedback, **values)
            v = read_json(f)
        finally:
            restore(run.ws, head)       # also drops anything the reviewer committed
        if (isinstance(v, dict) and v.get("verdict") in allowed
                and isinstance(v.get("issues"), list)
                and all(isinstance(i, str) and i.strip() for i in v["issues"])
                and (v["verdict"] == "approve" or v["issues"])):
            return {"verdict": v["verdict"], "issues": v["issues"]}
        feedback = (f"Write research/review.json with verdict in {allowed} and an issues array "
                    "of non-empty strings. A non-approve verdict requires at least one issue.")
    raise ContractError(f"{name}: no valid review after two attempts; experiment was not accepted")


def plan_problem(path):
    if not path.is_file():
        return "research/PLAN.md is missing"
    text = path.read_text()
    if not re.match(r"^# Experiment \d+: .+", text):
        return "plan must start with '# Experiment N: <title>'"
    sections = re.split(r"(?m)^## (.+)\s*$", text)
    headings = sections[1::2]
    if headings[:len(PLAN_SECTIONS)] != list(PLAN_SECTIONS):
        return "plan must contain these sections in order: " + ", ".join(PLAN_SECTIONS)
    if any(not body.strip() for body in sections[2:2 * len(PLAN_SECTIONS) + 1:2]):
        return "every required plan section must contain substantive text"
    if headings[len(PLAN_SECTIONS):] not in ([], ["Deviations"]):
        return "only an optional Deviations section may follow the required sections"
    return None


# ---------------------------------------------------------------- history

def write_history(run):
    """Regenerate research/HISTORY.md and research/LESSONS.md from the experiment records."""
    hist = ["# Experiment history", ""]
    lessons = ["# Lessons from earlier experiments", ""]
    if not run.history:
        hist.append("No experiments yet.")
        lessons.append("None yet.")
    else:
        hist += ["| # | Decision | Title | Val score | Gain | Reason |", "|---|---|---|---|---|---|"]
        for r in run.history:
            v = r.get("val") or {}
            g = f"{v['gain']:+.1%}" if v.get("gain") is not None else ""
            hist.append(f"| {r['exp']} | {r['decision']} | {r['title']} | {num(v.get('score'))} | {g} "
                        f"| {r['reason'].splitlines()[0][:120]} |")
        for r in run.history:
            a = r.get("analysis") or {}
            hist += ["", f"## Experiment {r['exp']}: {r['title']} ({r['decision']})", "",
                     f"Hypothesis: {r['hypothesis']}", "", f"Outcome: {r['reason']}"]
            if a.get("explanation"):
                hist += ["", f"Analysis: {a['explanation']}"]
            if a.get("next"):
                hist += ["", f"Suggested next: {a['next']}"]
            if a.get("lesson"):
                lessons.append(f"- (experiment {r['exp']}, {r['decision']}) {a['lesson']}")
    research = run.ws / "research"
    research.mkdir(exist_ok=True)
    (research / "HISTORY.md").write_text("\n".join(hist) + "\n")
    (research / "LESSONS.md").write_text("\n".join(lessons) + "\n")


def plan_summary(text):
    """Title (first heading) and hypothesis (the section named Hypothesis) of a plan."""
    title, hyp, inside = "", [], False
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if not title:
                title = heading
            inside = heading.lower().startswith("hypothesis")
        elif inside and line.strip():
            hyp.append(line.strip())
    return title.replace("|", "/")[:100], " ".join(hyp)[:600]


# ---------------------------------------------------------------- experiment loop

def goal_text(run):
    st, rep, task = run.state, run.state["check"], run.task
    lines = [f"The score is {'higher' if task['direction'] == 'maximize' else 'lower'}-is-better."]
    lines.append("Public web research is " + ("enabled." if task["agent"]["web_search"] else "disabled."))
    if rep["baseline"] == "reference":
        lines.append(f"Reference to beat (its implementation is hidden): val score {num(rep['scores'].get('val'))}.")
    if st["has_best"]:
        lines.append(f"Current best val score: {num(st['best_score'])}. A candidate is kept only if it is "
                     f"valid and beats the current best by more than {rep['threshold']:.1%} on the hidden val split.")
    else:
        lines.append("There is no valid solution yet. The first valid candidate becomes the current best.")
    return "\n".join(lines)


def run_tests(run):
    """Step 5: build, the task's test command, and the evaluator on train must all pass."""
    task, ws = run.task, run.ws
    ok, out = build(task, ws)
    if not ok:
        return False, "BUILD FAILED\n" + tail(out, 60, 4000)
    if task["commands"].get("test"):
        rc, out, _ = sh(task_command(task, "test"), cwd=ws, env=env_for(task, "eval"),
                        timeout=task["eval"]["timeout_s"])
        if rc:
            return False, f"TEST COMMAND FAILED (exit code {rc})\n" + tail(out, 60, 4000)
    r = evaluate(task, ws, task["p"]["train"])
    if not r["valid"]:
        return False, (f"THE EVALUATOR REJECTED THE CANDIDATE ON THE TRAIN SPLIT\n"
                       f"error: {r['error']}\nmetrics: {json.dumps(r['metrics'])}")
    return True, f"passed; train score {num(r['score'])}, metrics {json.dumps(r['metrics'])}"


def compare(run, sha):
    """Step 6: measure the candidate against the current best on val, from clean checkouts."""
    st, task = run.state, run.task
    best, cand = run.dir / "worktrees" / "best", run.dir / "worktrees" / "cand"
    res = {"valid": False, "score": None, "scores": [], "metrics": {}, "error": None,
           "best_score": None, "best_scores": [], "gain": None}
    checkout_clean(cand, sha)
    ok, out = build(task, cand)
    if not ok:
        res["error"] = "build failed in a clean checkout of the candidate:\n" + tail(out, 20, 1500)
        return res
    cands = {}
    if st["has_best"]:
        if st.get("best_built") != st["best_sha"]:
            checkout_clean(best, st["best_sha"])
            build(task, best)
            st["best_built"] = st["best_sha"]
        cands["best"] = (best, "eval")
    cands["cand"] = (cand, "eval")
    m = measure(task, cands, task["p"]["val"], task["eval"]["repetitions"])
    c, b = m["cand"], m.get("best")
    res.update(valid=c["valid"], score=c["median"], scores=c["scores"], metrics=c["metrics"], error=c["error"])
    if b:
        res.update(best_score=b["median"], best_scores=b["scores"])
        ref = b["median"]
        if ref is None:
            ref = res["best_score"] = st["best_score"]
            log(f"  warning: the current best was not valid in this measurement; "
                f"comparing with its recorded score {num(ref)}")
        if c["median"] is not None and ref is not None:
            res["gain"] = gain(task, c["median"], ref)
    return res


def result_text(run, rec):
    v, lines = rec.get("val"), [f"Decision: {rec['decision'].upper()}", f"Reason: {rec['reason']}"]
    if v:
        lines += [f"Valid on val: {v['valid']}", f"Candidate val score (median): {num(v['score'])}",
                  f"Current best val score, measured alongside: {num(v['best_score'])}"]
        if v["gain"] is not None:
            lines.append(f"Relative gain: {v['gain']:+.2%} (needed: more than {run.state['check']['threshold']:.1%})")
        lines.append(f"Aggregate metrics: {json.dumps(v['metrics'])}")
        if v["error"]:
            lines.append(f"Error: {v['error']}")
    else:
        lines.append("The candidate was not evaluated on val.")
    return "\n".join(lines)


def experiment(run, n):
    st, task, ws, b = run.state, run.task, run.ws, run.budget
    base = st["best_sha"]
    xd = run.dir / "experiments" / f"{n:04d}"
    shutil.rmtree(xd, ignore_errors=True)
    xd.mkdir(parents=True)
    restore(ws, base)
    write_history(run)
    plan = ws / "research" / "PLAN.md"
    plan.unlink(missing_ok=True)
    stop_file = ws / "research" / "stop.json"
    stop_file.unlink(missing_ok=True)
    rec = {"exp": n, "decision": "discard", "reason": "", "title": "", "hypothesis": "", "val": None,
           "analysis": {}, "sha": None, "tag": None, "agent_failed": False,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    common = {"base": base, "goal": goal_text(run)}

    def work(name, **values):
        """A step of the main agent, then commit its work and enforce the editable set."""
        ok = run_agent(run, n, name, **common, **values)
        rec["agent_failed"] |= not ok
        snapshot(ws)
        enforce_editable(run, base)

    def plan_step(name, **values):
        feedback = ""
        for _ in range(2):
            work(name, feedback=feedback, **values)
            request = read_json(stop_file)
            if name.startswith("propose") and stop_file.exists():
                if (not plan.exists() and st["has_best"] and isinstance(request, dict)
                        and all(isinstance(request.get(k), str) and request[k].strip()
                                for k in ("reason", "evidence", "alternatives"))):
                    return request
                feedback = ("Write either a plan or a convergence request, never both. A convergence "
                            "request requires a valid current best and non-empty reason, evidence, "
                            "and alternatives strings in research/stop.json.")
            else:
                feedback = plan_problem(plan)
                if feedback is None:
                    return None
        raise ContractError(f"{name}: invalid planning output after two attempts: {feedback}")

    def stages():
        """Steps 1 to 5. Returns None when the candidate is ready for val, else the reason."""
        propose = "propose_next" if run.history else "propose_first"
        log("  step 1: propose")
        request = plan_step(propose)
        if request is not None:
            write_json(xd / "stop.json", request)
            v = review(run, n, "review_stop", **common, stop_request=json.dumps(request, indent=2))
            write_json(xd / "stop_review.json", v)
            rec["title"] = "No worthwhile next experiment"
            if v["verdict"] == "approve":
                rec["decision"] = "stop"
                return "convergence approved: " + request["reason"]
            return "convergence request not approved: " + "; ".join(v["issues"])
        rec["title"], rec["hypothesis"] = plan_summary(plan.read_text())
        log(f"  plan: {rec['title']}")

        log("  step 2: review plan")
        reviews = []
        for round_no in range(b["max_plan_revisions"] + 1):
            v = review(run, n, "review_plan", **common)
            reviews.append(v)
            log(f"  plan review: {v['verdict']}")
            if v["verdict"] != "revise" or round_no == b["max_plan_revisions"]:
                break
            plan_step("revise_plan", issues="\n".join(f"- {i}" for i in v["issues"]))
        write_json(xd / "plan_review.json", reviews)
        if plan.exists():
            shutil.copy(plan, xd / "plan.md")
            rec["title"], rec["hypothesis"] = plan_summary(plan.read_text())
        if v["verdict"] == "reject":
            return "plan rejected by review: " + "; ".join(v["issues"])[:300]
        if v["verdict"] != "approve":
            return "plan revisions exhausted without approval: " + "; ".join(v["issues"])[:300]

        log("  step 3: implement")
        work("implement")
        if not code_changed(ws, base):
            return "no code change"

        log("  step 4: review code")
        v = review(run, n, "review_code", **common)
        code_reviews = [v]
        write_json(xd / "code_review.json", code_reviews)
        log(f"  code review: {v['verdict']}, {len(v['issues'])} issue(s)")
        if v["verdict"] != "approve" and v["issues"]:
            work("fix_review", issues="\n".join(f"- {i}" for i in v["issues"]))
            v = review(run, n, "review_code", **common)
            code_reviews.append(v)
            write_json(xd / "code_review.json", code_reviews)
            if v["verdict"] != "approve":
                return "code review still requires changes: " + "; ".join(v["issues"])[:300]

        log("  step 5: test")
        for attempt in range(b["max_fix_attempts"] + 1):
            ok, out = run_tests(run)
            with open(xd / "test.log", "a") as f:
                f.write(f"--- attempt {attempt + 1}: {'pass' if ok else 'fail'}\n{out}\n\n")
            if ok:
                return None
            log(f"  tests failed (attempt {attempt + 1})")
            if attempt < b["max_fix_attempts"]:
                work("fix_tests", log=out)
        return f"tests still failing after {b['max_fix_attempts']} fix attempt(s)"

    reason = stages()
    if plan.is_file():
        shutil.copy(plan, xd / "plan.md")
    plan_text = (xd / "plan.md").read_text() if (xd / "plan.md").exists() else \
        json.dumps(read_json(xd / "stop.json", {}), indent=2)
    sha = squash(ws, base, f"exp {n:04d}: {rec['title'] or 'untitled'}")
    rec["sha"] = sha
    if sha:
        (xd / "diff.patch").write_text(git(ws, "diff", base, sha, "--", ".", ":(exclude)research") + "\n")

    if reason is None:
        log("  step 6: evaluate on val")
        v = rec["val"] = compare(run, sha)
        write_json(xd / "eval.json", v)
        threshold = st["check"]["threshold"]
        if not v["valid"]:
            reason = f"not valid on val: {v['error']}"
        elif v["gain"] is None:
            rec["decision"], reason = "keep", "first valid solution" if not st["has_best"] else \
                "valid, and the current best could not be measured"
        elif v["gain"] > threshold:
            rec["decision"], reason = "keep", f"gain {v['gain']:+.1%} is above the threshold {threshold:.1%}"
        else:
            reason = f"gain {v['gain']:+.1%} is not above the threshold {threshold:.1%}"
        if rec["decision"] == "keep" and task["p"]["guard"]:
            log("  step 6b: guard split (pass/fail)")
            g = evaluate(task, run.dir / "worktrees" / "cand", task["p"]["guard"])
            v["guard_valid"] = g["valid"]
            if not g["valid"]:
                rec["decision"], reason = "discard", GUARD_FAILED
    rec["reason"] = reason
    log(f"  step 7: {rec['decision'].upper()} ({reason.splitlines()[0][:100]})")

    if sha:
        rec["tag"] = f"exp-{n:04d}-{rec['decision']}"
        git(ws, "tag", "-f", rec["tag"], sha)
    next_best = sha if rec["decision"] == "keep" else base
    if rec["decision"] != "keep":
        restore(ws, base)

    if plan_text:
        log("  step 8: analyze")
        analysis = ws / "research" / "analysis.json"
        feedback = ""
        for _ in range(2):
            analysis.unlink(missing_ok=True)
            try:
                run_agent(run, n, "analyze", **common, result=result_text(run, rec),
                          plan_text=plan_text, tag=rec["tag"] or "(nothing was committed)",
                          feedback=feedback)
                a = read_json(analysis)
            finally:
                restore(ws, next_best)
            if (isinstance(a, dict) and a.get("supported") in ("yes", "no", "unclear")
                    and all(isinstance(a.get(k), str) and a[k].strip()
                            for k in ("explanation", "lesson", "next"))):
                break
            feedback = ("Write research/analysis.json with supported equal to yes, no, or unclear, "
                        "and non-empty explanation, lesson, and next strings.")
        else:
            raise ContractError("analyze: no valid analysis after two attempts")
        rec["analysis"] = a
        write_json(xd / "analysis.json", rec["analysis"])
        restore(ws, next_best)

    rec["seconds"] = round(time.time() - t0)
    state_after = {"done": n}
    if rec["decision"] == "keep":
        state_after.update(best_sha=sha, has_best=True, best_score=rec["val"]["score"])
    if rec["decision"] == "stop":
        state_after["stop_reason"] = reason
    rec["state_after"] = state_after
    run.history.append(rec)
    journal_tmp = run.records_file.with_suffix(".jsonl.tmp")
    journal_tmp.write_text("".join(json.dumps(r) + "\n" for r in run.history))
    journal_tmp.replace(run.records_file)
    write_history(run)
    st.update(state_after)
    run.save()


def loop(run):
    st, b = run.state, run.budget
    if st.get("final"):
        raise Fail("this run already used the test split; no more experiments are allowed")
    st["status"] = "running"
    run.save()
    log(f"run {st['run_id']}: agent={st['agent']}, budget {b['max_experiments']} experiments "
        f"or {b['max_hours']} hours")
    log(f"workspace: {run.ws}")
    while (not st.get("stop_reason") and st["done"] < b["max_experiments"]
           and run.elapsed() < b["max_hours"] * 3600):
        n = st["done"] + 1
        log(f"experiment {n}")
        experiment(run, n)
    st["status"] = "converged" if st.get("stop_reason") else "finished"
    run.save()


# ---------------------------------------------------------------- final and report

def final(run):
    """Evaluate the reference, the starting target and the final best on the test split, once."""
    st, task, rep = run.state, run.task, run.state["check"]
    if st.get("final"):
        raise Fail("the test split was already used for this run; see report.md")
    if not task["p"]["test"]:
        raise Fail("the test split is not available; run `final` again when it is")
    reps = task["eval"]["repetitions"]
    first_valid = next((r for r in run.history if r["decision"] == "keep"), None)
    with tempfile.TemporaryDirectory() as tmp:
        cands = {}
        if task["p"]["reference"]:
            ref = Path(shutil.copytree(task["p"]["reference"], Path(tmp) / "reference"))
            if build(task, ref, "ref")[0]:
                cands["reference"] = (ref, "ref")
        if rep["state"] == "existing":
            start = run.dir / "worktrees" / "cand"
            checkout_clean(start, "start")
            if build(task, start)[0]:
                cands["start"] = (start, "eval")
        elif first_valid and first_valid["sha"] != st["best_sha"]:
            first = run.dir / "worktrees" / "cand"
            checkout_clean(first, first_valid["sha"])
            if not build(task, first)[0]:
                raise Fail("the first valid candidate could not be rebuilt for final comparison")
            cands["first_valid"] = (first, "eval")
        if st["has_best"]:
            best = run.dir / "worktrees" / "best"
            checkout_clean(best, st["best_sha"])
            st["best_built"] = st["best_sha"] if build(task, best)[0] else None
            cands["best"] = (best, "eval")
        log(f"final: evaluating {', '.join(cands) or 'nothing'} on the test split, {reps} runs each")
        m = measure(task, cands, task["p"]["test"], reps)
    if rep["state"] == "scratch" and first_valid and first_valid["sha"] == st["best_sha"] and "best" in m:
        # Same code: reuse its measurements rather than scoring an identical copy twice.
        m["first_valid"] = m["best"]
    st["final"] = {n: {k: r[k] for k in ("valid", "median", "scores", "metrics", "error")}
                   for n, r in m.items()}
    st["final"]["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run.save()


def report(run):
    st, rep, task = run.state, run.state["check"], run.task
    kept = [r for r in run.history if r["decision"] == "keep"]
    out = [f"# Run {st['run_id']}", "",
           f"- Task: {task['name']} ({task['direction']})",
           f"- Agent: {st['agent']}",
           f"- Status: {st['status']}, {st['done']} experiment(s), {len(kept)} kept",
           f"- Time: {st.get('elapsed_s', 0) / 3600:.2f} hours",
           (f"- Agent cost reported: ${st.get('cost_usd', 0):.2f}" if run.agent.get("cost_key")
            else "- Agent cost: not reported by this agent CLI"),
           f"- Start: {rep['state']}; baseline: {rep['baseline'] or 'none'}, "
           f"val score {num(rep['scores'].get('val'))}",
           f"- Acceptance threshold: {rep['threshold']:.1%} (noise {num(rep['noise'])})",
           f"- Best val score: {num(st['best_score']) if st['has_best'] else 'no valid solution'}",
           "", "## Experiments", "",
           "| # | Decision | Title | Val score | Gain | Reason |", "|---|---|---|---|---|---|"]
    for r in run.history:
        v = r.get("val") or {}
        g = f"{v['gain']:+.1%}" if v.get("gain") is not None else ""
        out.append(f"| {r['exp']} | {r['decision']} | {r['title']} | {num(v.get('score'))} | {g} "
                   f"| {r['reason'].splitlines()[0][:100]} |")
    out += ["", "## Best val score over time", ""]
    out += [f"- experiment {r['exp']}: {num(r['val']['score'])}" for r in kept] or ["- no kept experiment"]
    fin = st.get("final")
    if fin:
        out += ["", "## Test split (run once)", "", "| Candidate | Valid | Test score (median) |", "|---|---|---|"]
        for name in ("reference", "start", "first_valid", "best"):
            if name in fin:
                out.append(f"| {name} | {fin[name]['valid']} | {num(fin[name]['median'])} |")
        base = "reference" if "reference" in fin else "start"
        if base in fin and not fin[base]["valid"]:
            out += ["", f"The {base} is not valid on test ({fin[base]['error']}), so no speed ratio "
                    "against it is given. A valid best beats it on correctness."]
        val_base = rep["scores"].get("val")
        if all(k in fin and fin[k]["median"] for k in (base, "best")) and val_base and st["best_score"]:
            def factor(new, old):       # above 1 means better, for either direction
                return new / old if task["direction"] == "maximize" else old / new
            val_f = factor(st["best_score"], val_base)
            test_f = factor(fin["best"]["median"], fin[base]["median"])
            out += ["", f"Improvement of the best over the {base}: {val_f:.3g}x on val, {test_f:.3g}x on test.",
                    "Different factors can reflect input distributions and baseline scaling as well as "
                    "overfitting. Compare the task's workloads before drawing a causal conclusion."]
        if all(k in fin and fin[k].get("valid") and fin[k].get("median") for k in ("first_valid", "best")):
            first_score, best_score = fin["first_valid"]["median"], fin["best"]["median"]
            progress = best_score / first_score if task["direction"] == "maximize" else first_score / best_score
            out += ["", f"Progress from the first valid solution: {progress:.3g}x on test."]
    else:
        out += ["", "The test split has not been evaluated yet."]
    out += ["", "## Inspecting the run", "",
            f"- Workspace (git): `{run.ws}`",
            "- `git log best` shows the kept experiments; `git tag` lists every experiment.",
            f"- Per-experiment files: `{run.dir / 'experiments'}`",
            f"- Agent transcripts: `{run.dir / 'logs'}`"]
    text = "\n".join(out) + "\n"
    (run.dir / "report.md").write_text(text)
    return text


# ---------------------------------------------------------------- command line

def start(run):
    try:
        loop(run)
        if run.task["p"]["test"]:
            final(run)
        else:
            log("the test split is not available; run `final` when it is")
    except KeyboardInterrupt:
        restore(run.ws, run.state["best_sha"])
        run.state["status"] = "paused"
        run.save()
        report(run)
        raise Fail(f"interrupted. Continue with: python hillclimb.py resume {run.dir}")
    except (AgentDown, ContractError) as e:
        restore(run.ws, run.state["best_sha"])
        run.state["status"] = "paused"
        run.save()
        report(run)
        raise Fail(f"{e}\n\nRun paused; the unfinished experiment is not recorded and the test split "
                   f"is untouched. Continue with: python hillclimb.py resume {run.dir}")
    print()
    print(report(run))
    print(f"report: {run.dir / 'report.md'}")


def main(argv=None):
    load_dotenv(HARNESS / ".env")
    ap = argparse.ArgumentParser(description="Hill-climbing harness")
    ap.add_argument("--runs-dir", default=str(HOME / "runs"), help="where run directories live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate a task folder")
    c.add_argument("task")
    c.add_argument("--noise-runs", type=int, default=5)
    r = sub.add_parser("run", help="start a new run")
    r.add_argument("task")
    r.add_argument("--agent", default="cursor")
    r.add_argument("--agents-file", default=str(HARNESS / "agents.toml"))
    for name in ("resume", "status", "report", "final", "dev-eval"):
        s = sub.add_parser(name)
        s.add_argument("run", nargs="?")
        if name == "resume":
            r2 = s
    for p in (r, r2):
        p.add_argument("--max-experiments", type=int)
        p.add_argument("--max-hours", type=float)
    a = ap.parse_args(argv)

    def budget():
        return {k: v for k, v in (("max_experiments", a.max_experiments), ("max_hours", a.max_hours))
                if v is not None}

    if a.cmd == "check":
        return 0 if check(a.task, a.noise_runs)["ok"] else 1
    if a.cmd == "run":
        task = load_task(a.task)
        rep = read_json(task["dir"] / "check_report.json")
        if task["problems"] or not rep or not rep["ok"] or rep["task_hash"] != task_hash(task):
            raise Fail(f"the task has no passing, current check. Run: python hillclimb.py check {a.task}")
        agents_file = Path(a.agents_file).resolve()
        if a.agent not in tomllib.loads(agents_file.read_text()):
            raise Fail(f"agent '{a.agent}' is not defined in {agents_file}")
        start(new_run(task, rep, a.agent, agents_file, a.runs_dir, budget()))
        return 0
    run = find_run(a.run, a.runs_dir)
    if a.cmd == "resume":
        run.state["budget"] = {**run.state.get("budget", {}), **budget()}
        run.budget.update(run.state["budget"])
        start(run)
    elif a.cmd == "final":
        final(run)
        print(report(run))
    elif a.cmd == "report":
        print(report(run))
    elif a.cmd == "status":
        st = run.state
        cost = f"${st.get('cost_usd', 0):.2f}" if run.agent.get("cost_key") else "cost not reported"
        print(f"{st['run_id']}: {st['status']}, {st['done']} experiment(s) done, "
              f"best val {num(st['best_score']) if st['has_best'] else 'none'}, "
              f"{st.get('elapsed_s', 0) / 3600:.2f} h, {cost}")
        for rec in run.history[-10:]:
            print(f"  {rec['exp']:>3}  {rec['decision']:<8} {rec['title'][:50]:<50}  {rec['reason'].splitlines()[0][:60]}")
    elif a.cmd == "dev-eval":
        ok, out = build(run.task, run.ws)
        res = evaluate(run.task, run.ws, run.task["p"]["train"]) if ok else \
            {"valid": False, "score": None, "metrics": {}, "error": "build failed:\n" + tail(out)}
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
