"""Verification pack: Mike's review of explained genomes, from the export JSONs alone.

For each genome (in the pre-registered order, never re-sorted) the review PDF has a
verification checklist block, ported from ``experiments/10seed/verification/make_packets.py``:

* recorded vs live checks: the agent's report, the orchestrator's ``verified_by_orchestrator``
  block, and the export's live values (current code);
* "what to look at": checks that disagree, the largest closed-form errors, strong claims or
  rare regimes in leaf hypotheses, hidden-node splits, consolidations, pre-v3 tooling, and the
  root's hypothesis against its closed form;
* blank verdict fields, and the one-line command that records the verdict as
  ``mike_verification`` in ``experiments/10seed/explanations/<label>.json`` (read by
  ``build_explanations_csv.py``, and by the next export as the verifier).

Then the genome's appendix follows. Printed text never names the drafting assistant.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import render_tex
from .text import fnum, printed, tex, tex_code, tex_num

STRONG = [(3, r"\bexact(ly)?\b|\bbit-identical\b|\b(AND|OR|XOR|NOT)\b|\boverride\b|\brare\b"),
          (2, r"\bonly\b|\bnever\b|\balways\b|\bthreshold\b|\bstep\b|\bdead\b|\binert\b|\bvestig\w*|\bsaturat\w*"),
          (2, r"\b\d+\s*/\s*\d+\b|\b\d+\s+(train(ing)?\s+)?(rows|patients|customers|cases)\b|\bn_on\b"),
          (1, r"\bswitch\w*|\bfires?\b|\bgate[sd]?\b|\bmonoton\w*|\bperfect(ly)?\b|\bidentical\b")]


def strong_claim(text, width=230) -> Tuple[int, Optional[str]]:
    """(score, snippet): the hypothesis's most assertive sentence (make_packets.py)."""
    if not text:
        return 0, None
    best: Tuple[int, Optional[str]] = (0, None)
    for sent in re.split(r"(?<=[.;])\s+", text):
        score, first = 0, None
        for w, pat in STRONG:
            for m in re.finditer(pat, sent):
                score += w
                first = m.start() if first is None else min(first, m.start())
        if score > best[0]:
            lo = max(0, first - width // 3)
            snip = sent[lo:lo + width].strip()
            best = (score, ("..." if lo else "") + snip + ("..." if lo + width < len(sent) else ""))
    return best


def min_n_on(a) -> Optional[float]:
    best = None
    for e in a["evidence"]:
        stack = [e["result"].get("payload")]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("n_on", "n_active", "n_rows_on", "n_fire", "n_firing") and isinstance(v, (int, float)) \
                            and not isinstance(v, bool) and v > 0:
                        best = v if best is None else min(best, v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(i for i in x if isinstance(i, (dict, list)))
    return best


def _fagree(a, b, rounded=False) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if abs(a - b) <= 1e-12:
        return True
    if rounded:
        return abs(a - b) <= 5e-3 * max(abs(b), 1e-12) + 1e-12
    # live values are stored at 6 significant figures
    return abs(a - b) <= 5e-6 * max(abs(b), 1e-12)


def live_values(doc) -> Dict[str, Any]:
    leaves = [a for a in doc["annotations"] if a["kind"] == "leaf"]
    cfe = [(a["closed_form"].get("error") or {}).get("closed_form_max_abs_err") for a in leaves]
    num = [(a["closed_form"].get("error") or {}).get("numeric_max_abs_err") for a in leaves]
    cfe = [x for x in cfe if x is not None]
    num = [x for x in num if x is not None]
    h = doc["coverage"]["report"].get("hierarchy") or {}
    s = doc["coverage"]["summary"]
    fp = doc["function_preservation"]["max_abs_err"]
    return {
        "function_preservation_max_abs_err": fp,
        "closed_form_max_abs_err": max(cfe) if cfe else None,
        "numeric_max_abs_err": max(num) if num else None,
        "structural_coverage": s["structural_coverage"],
        "compositional_coverage": s["compositional_coverage"],
        "hierarchy_complete": bool(h.get("complete")),
        "roots": list(h.get("roots") or doc["hierarchy"]["roots"]),
        "leaf_count": s["leaves"], "composition_count": s["compositions"],
        "op_counts": doc["operations"]["counts"],
        "full_coverage": bool(s["full_coverage"] and fp is not None and fp < 1e-9),
        "leaves": sorted(((a["name"], (a["closed_form"].get("error") or {}).get("closed_form_max_abs_err"),
                           (a["closed_form"].get("error") or {}).get("numeric_max_abs_err")) for a in leaves),
                         key=lambda t: -(t[1] if t[1] is not None else 1e9)),
    }


def _ops(counts) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in sorted(counts.items()))


def check_rows(doc) -> Tuple[List[List[str]], List[str]]:
    """[[check, agent, orchestrator, live, status]] and the names that disagree."""
    rec = doc.get("record") or {}
    v = rec.get("verified_by_orchestrator") or {}
    hv = v.get("hierarchy") or {}
    L = live_values(doc)
    rows, bad = [], []

    def row(name, agent, orch, live, ok):
        rows.append([name, agent, orch, live, "agree" if ok else "DISAGREE"])
        if not ok:
            bad.append(name)
    fps = (rec.get("function_preservation_max_abs_err"), v.get("function_preservation_max_abs_err"),
           L["function_preservation_max_abs_err"])
    row("Function preservation, max |change in output|", fnum(fps[0]), fnum(fps[1]), fnum(fps[2]),
        all(x is not None and x < 1e-9 for x in fps))
    row("Closed-form max abs error (all leaves)", fnum(rec.get("closed_form_max_abs_err"), 4),
        fnum(v.get("closed_form_max_abs_err"), 4), fnum(L["closed_form_max_abs_err"], 4),
        _fagree(v.get("closed_form_max_abs_err"), L["closed_form_max_abs_err"])
        and _fagree(rec.get("closed_form_max_abs_err"), L["closed_form_max_abs_err"], rounded=True))
    row("Numeric-function max abs error", "n/a", fnum(v.get("numeric_max_abs_err")), fnum(L["numeric_max_abs_err"]),
        (v.get("numeric_max_abs_err") or 0) < 1e-9 and (L["numeric_max_abs_err"] or 0) < 1e-9)
    row("Structural coverage", fnum(rec.get("structural_coverage")), fnum(v.get("structural_coverage")),
        fnum(L["structural_coverage"]),
        rec.get("structural_coverage") == v.get("structural_coverage") == L["structural_coverage"])
    row("Compositional coverage", fnum(rec.get("compositional_coverage")), fnum(v.get("compositional_coverage")),
        fnum(L["compositional_coverage"]),
        rec.get("compositional_coverage") == v.get("compositional_coverage") == L["compositional_coverage"])
    row("Hierarchy complete", fnum(rec.get("hierarchy_complete")), fnum(hv.get("complete")), fnum(L["hierarchy_complete"]),
        rec.get("hierarchy_complete") == hv.get("complete") == L["hierarchy_complete"])
    row("Root annotation", "n/a", ", ".join(hv.get("roots") or []), ", ".join(L["roots"]),
        list(hv.get("roots") or []) == L["roots"])
    row("Leaf / composition counts",
        f"{rec.get('n_leaf_annotations')} / {rec.get('n_composition_annotations')}", "n/a",
        f"{L['leaf_count']} / {L['composition_count']}",
        (rec.get("n_leaf_annotations"), rec.get("n_composition_annotations")) == (L["leaf_count"], L["composition_count"]))
    oc = {k: n for k, n in L["op_counts"].items()}
    row("Operation counts", f"{rec.get('n_splits')} splits, {rec.get('n_identity_nodes')} identity",
        _ops(v.get("op_counts")), _ops(oc),
        v.get("op_counts") == oc and rec.get("n_splits") == oc.get("split_node", 0)
        and rec.get("n_identity_nodes") == oc.get("add_identity_node", 0))
    row("Full coverage", fnum(rec.get("full_coverage")), fnum(v.get("full_coverage")), fnum(L["full_coverage"]),
        rec.get("full_coverage") == v.get("full_coverage") == L["full_coverage"])
    return rows, bad


def what_to_look_at(doc, disagreements: List[str]) -> List[str]:
    """Printed (LaTeX) bullet texts, at most 6."""
    look: List[str] = []
    L = live_values(doc)
    rec = doc.get("record") or {}
    n_train = doc["function_preservation"].get("n") or 0
    if disagreements:
        look.append(r"\textbf{Checks disagree:} " + tex(", ".join(disagreements)) + ". Resolve this before verifying.")
    numeric = [t for t in L["leaves"] if t[1] is not None]
    if numeric:
        t0 = numeric[0]
        txt = "Largest closed-form error: " + tex_code(t0[0]) + f" ({tex(fnum(t0[1]))})"
        if len(numeric) > 1 and numeric[1][1] > 0.01:
            txt += ", then " + tex_code(numeric[1][0]) + f" ({tex(fnum(numeric[1][1]))})"
        txt += (". This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric "
                "function is exact. " + ("It is large enough that a reader using the displayed coefficients "
                                         "would get visibly different exit values: check which coefficient drives it."
                                         if t0[1] > 0.1 else
                                         "Check that the rounded formula still supports the hypothesis as worded."))
        look.append(txt)
    cands, seen = [], set()
    for a in doc["annotations"]:
        if a["kind"] != "leaf":
            continue
        h = a["hypothesis"] or ""
        key = re.sub(r"\b\w+_[a-z]\b", "", h)[:300]
        if key in seen:
            continue
        seen.add(key)
        n_on = min_n_on(a)
        score, snip = strong_claim(h)
        rare = n_on is not None and n_on < max(30, 0.01 * n_train)
        if rare or score >= 3:
            cands.append((0 if rare else 1, n_on if rare else -score, a["name"], snip, n_on if rare else None))
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, name, snip, n_on in cands[:2]:
        t = "Strong claim in " + tex_code(name) + ": "
        if n_on is not None:
            t += f"its evidence includes a category or regime active on only {int(n_on)} training rows. "
        if snip:
            t += r"\emph{``" + tex(snip) + "''}"
        t += " Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?"
        look.append(t)
    ops = doc["operations"]["refactoring"]
    inputs = {n["id"] for n in doc["phenotype"]["nodes"] if n["type"] == "input"}

    def base(n):
        return re.sub(r"(_[a-z])+$", "", str(n))
    hidden_splits = [o for o in ops if o["type"] == "split_node" and base((o["params"] or {}).get("node_id")) not in inputs]
    n_splits = sum(1 for o in ops if o["type"] == "split_node")
    if hidden_splits:
        look.append(f"{len(hidden_splits)} hidden-node split(s), which decide how a unit's roles are divided: "
                    + "; ".join(tex_code((o["params"] or {}).get("node_id")) + r"$\to$"
                                + ", ".join(tex_code(c) for c in ((o.get("result") or {}).get("created_nodes") or []))
                                for o in hidden_splits[:3])
                    + ". Do the copies really play the different roles the annotations give them?")
    elif n_splits:
        look.append(f"{n_splits} split(s), all of input nodes (an input feeding two annotations). These are routine; "
                    "check only that each input copy sits in the annotation whose hypothesis uses it.")
    else:
        look.append("No splits: every node sits in one annotation as evolved.")
    nc = doc["operations"]["counts"].get("consolidate_node")
    if nc:
        look.append(f"{nc} consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.")
    if rec and rec.get("protocol_version") != "v3":
        look.append("Explained with protocol v2 tooling" + (" (the pilot)" if rec.get("pilot") else "")
                    + ": the formula tool then showed the output node as ReLU, while the model applies sigmoid. "
                      "Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix "
                      "(the closed forms here use sigmoid). Check the root's wording.")
    look.append("Read the root's hypothesis against its closed form: does the stated combination of children "
                "(additive, gated, serial) match the formula?")
    return look[:6]


def records_dir_shown() -> str:
    """The records folder in use, repository-relative (``EXPLANEAT_RECORDS_DIR`` or the default)."""
    from ..analysis.explanation_records import record_set
    return record_set()["records_dir"]


def records_opt() -> str:
    """`` --records-dir <dir>`` when the records are not the repository's default folder."""
    rd = records_dir_shown()
    return "" if rd == "experiments/10seed/explanations" else f" --records-dir {rd}"


def record_command(label: str) -> str:
    return (f"uv run python -m explaneat export record-verdict {label}{records_opt()} --verified "
            f"--minutes N --changes \"...\" --notes \"...\"")


def errata_line(rec) -> list:
    """One line naming the errata applied to the record (its ``errata`` list), if any."""
    errata = (rec or {}).get("errata") or []
    if not errata:
        return []
    dates = sorted({str(e.get("date")) for e in errata if e.get("date")})
    refs = sorted({str(e.get("ref")) for e in errata if e.get("ref")})
    return [r"\textbf{Errata applied " + tex(", ".join(dates)) + r"}: " + tex(f"{len(errata)} correction(s)")
            + (r", see \texttt{" + tex(", ".join(refs)) + "}" if refs else "") + ".", ""]


VERDICT_BOXES = r"$\square$\,ok\ \ $\square$\,wrong\ \ $\square$\,unsure"
AP = "A'"
BADGE_WORD = {"C": "computed", "R": "rechecked", "U": "unchecked", "F": "FAILED", "n/a": "n/a"}


def _short(s: str, n: int = 260) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def _queue_rows(items, start: int = 1) -> List[str]:
    rows = []
    for k, it in enumerate(items, start):
        head = (r"\texttt{" + tex(it["sid"]) + "}" + r" \quad " + tex(f"[{it['tier']}] {BADGE_WORD.get(it['badge'], it['badge'])}"
                f", score {it['score']:g}") + r" \quad {\scriptsize " + tex_code(_short(it["ref"], 70)) + "}")
        cell = [head, r"\newline ``" + tex(_short(it["text"])) + "''"]
        if it["why"]:
            cell.append(r"\newline {\scriptsize Why: " + tex("; ".join(it["why"])) + "}")
        for f in it["verifier"][:2]:
            cell.append(r"\newline {\scriptsize\textbf{Verifier} (" + tex(f.get("source")) + "): "
                        + tex(f"{f.get('grade')} {f.get('level') or ''}") + ("; stale" if f.get("stale") else "")
                        + ". " + tex(_short(f.get("evidence"), 220)) + "}")
        done = it.get("verdict")
        box = (r"\textbf{" + tex(done) + "}" if done and done != "stale"
               else (r"{\scriptsize changed since your verdict}\newline " if done == "stale" else "") + VERDICT_BOXES)
        rows.append(f"{k} & " + "".join(cell) + r" & {\scriptsize " + box + r"} \\ \midrule")
    return rows


def _queue_table(items, start=1) -> List[str]:
    return ([r"{\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{@{}>{\raggedleft\arraybackslash}p{0.5cm}>{\raggedright\arraybackslash}p{12.4cm}"
             r">{\raggedright\arraybackslash}p{2.6cm}@{}}",
             r"\toprule \# & Sentence (sid, tier, status, location) & Your verdict \\ \midrule \endhead"]
            + _queue_rows(items, start) + [r"\end{longtable}}", ""])


def review_queue_tex(rv: Dict[str, Any]) -> List[str]:
    """The signoff-v1 review queue for one genome (docs/design/human-cross-verification.md, 4 and 7)."""
    lab = rv["label"]
    b, t = rv["badges"], rv["tiers"]
    L = [r"\textbf{Review queue (signoff-v1).} Sentences ranked by the risk of the error mechanisms found in "
         r"the 2026-11 review: unchecked hand-typed numbers, scope words (``only'', ``all'', ``zero''), formulas "
         r"written as exact, small groups, direct vs whole-model effects, verifier findings, changes since your "
         r"last verdict. Status: " + tex(f"computed {b.get('C', 0)}, rechecked {b.get('R', 0)}, unchecked "
                                          f"{b.get('U', 0)}, failed {b.get('F', 0)}") +
         r" (no sentence can be ``computed'' until numbers carry provenance). Tiers: " +
         tex(f"gate {t.get('G', 0)}, A {t.get('A', 0)} (all reviewed), A-overflow {t.get(AP, 0)}, "
             f"B {t.get('B', 0)}, C {t.get('C', 0)} (sampled)") + ". "
         + tex(f"Verifier findings attached: {rv['overlay']['attached']}"
               + (f" ({rv['overlay']['stale']} stale)" if rv['overlay']['stale'] else "")) + ". "
         + r"\textbf{Estimated time: about " + tex(str(rv["estimate_minutes"])) + r" minutes} ("
         + tex(f"{rv['n_queue']} queue items, {rv['n_sample']} sampled") + ").", ""]
    if rv["changes"]:
        L += [r"\textbf{Changed since your last review.}", r"\begin{itemize}"]
        for c in rv["changes"][:12]:
            line = tex(f"{c['sid']} ({c['state']}; you said {c['verdict']}): ")
            if c["state"] == "changed":
                line += r"\emph{" + tex(_short(c.get("word_diff"), 300)) + "}"
                if c.get("numbers"):
                    line += tex(" Numbers: " + "; ".join(f"{a} -> {x}" for a, x in c["numbers"]) + ".")
            line += tex(" Why: " + "; ".join(c["why"]) + ".")
            L.append(r"\item " + line)
        L += [r"\end{itemize}"]
    elif not rv["n_verdicts"]:
        L += [r"{\small No sentence verdicts recorded yet, so nothing has changed since a review.\par}", ""]
    else:
        L += [r"{\small Nothing has changed since your last verdicts.\par}", ""]
    if rv["queue"]:
        L += [r"\textbf{Gate and tier-A items} (every one needs a verdict):", ""] + _queue_table(rv["queue"])
    sample = rv["sample"]["BC"] + rv["sample"]["A_prime"]
    if sample:
        L += [r"\textbf{Random sample} (drawn by code from tiers A-overflow, B and C; every one needs a verdict; "
              r"one ``wrong'' draws 10 more):", ""] + _queue_table(sample, len(rv["queue"]) + 1)
    for f in rv.get("unattached_verifier") or []:
        L += [r"{\small\textbf{Verifier, not about one sentence} (" + tex(f.get("location")) + "): "
              + tex(f"{f.get('grade')} {f.get('level') or ''}. {_short(f.get('evidence'), 240)}") + r"\par}"]
    L += [r"{\small\raggedright\textbf{Record a verdict} on an item (append-only, "
          + tex_code(f"experiments/10seed/review/{lab}.verdicts.jsonl") + r"):\par}", r"{\footnotesize\raggedright\texttt{" + tex(
              f"uv run python -m explaneat review verdict {lab}{records_opt()} <sid> confirmed|wrong|unsure --note \"...\"")
          + r"}\par}",
          r"{\small Then check and record the sign-off (it refuses until every item above has a verdict):\par}",
          r"{\footnotesize\raggedright\texttt{" + tex(f"uv run python -m explaneat review signoff {lab}{records_opt()} --write --minutes N")
          + r"}\par}", ""]
    return L


def checklist_block(doc, order: int) -> str:
    lab = doc["label"]
    rows, bad = check_rows(doc)
    look = what_to_look_at(doc, bad)
    rec = doc.get("record") or {}
    hv = doc["authorship"]["human_verification"]
    L = [f"\\section*{{{order}. {tex(lab)}: verification checklist}}",
         f"\\addcontentsline{{toc}}{{section}}{{{order}. {tex(lab)}: verification checklist}}",
         f"\\label{{rev:{render_tex.label_key(lab)}}}",
         (r"\textbf{Already recorded:} " + ("verified" if hv.get("verified") else "not verified")
          + (f", {tex(fnum(hv.get('minutes')))} min" if hv.get("minutes") is not None else "")
          + (f", changes: {tex(hv.get('changes'))}" if hv.get("changes") else "") + "."
          if hv else r"\textbf{Already recorded:} nothing (no verdict in the record)."),
         "",
         *errata_line(rec),
         r"\textbf{Recorded vs live.} Agent = the drafting agent's report; orchestrator = the record's "
         r"automated re-check; live = this export (current code, training split).",
         "",
         r"{\footnotesize",
         r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.1cm}"
         r">{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.0cm}l@{}}",
         r"\toprule", r"Check & Agent & Orchestrator & Live & Status \\", r"\midrule"]
    for r in rows:
        st = r[4] if r[4] == "agree" else r"\textbf{DISAGREE}"
        L.append(" & ".join(tex(c) for c in r[:4]) + f" & {st} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}}", ""]
    if doc.get("review"):    # the risk-ranked queue replaces the "what to look at" bullets
        if bad:
            L += [look[0], ""]
        L += review_queue_tex(doc["review"])
    else:
        L += [r"\textbf{What to look at.}", r"\begin{itemize}"]
        L += [r"\item " + x for x in look]
        L += [r"\end{itemize}"]
    if rec.get("tool_bugs"):
        L += [r"\textbf{Tool notes (record):} " + tex(rec["tool_bugs"]), ""]
    pl = doc.get("plain_language")
    if pl:   # schema 1.1+: the plain-language layer, with the evidence of the claims it cites
        from .render_plain import _status, claims_table, sentences_tex, v5_summary
        d = pl.get("drafted") or {}
        v5 = bool(d.get("readdown"))
        summ = (v5_summary(d, pl) if v5 else d.get("summary")) or (pl.get("template") or {}).get("summary") or []
        from .render_plain import outcome_callout
        L += outcome_callout(pl)
        verdict_cmd = (r"\texttt{" + tex(f"uv run python -m explaneat export record-plain-verdict {lab} --verified "
                                          f"--minutes N --notes \"...\"") + "}")
        if v5:   # the read-down document that follows opens with the summary: do not repeat it here
            L += [r"{\raggedright\textbf{Plain-language layer} (" + tex(_status(pl)) + r"): the read-down document "
                  r"follows this page (summary, parts, evidence, worth checking; also \texttt{" + tex(f"{lab}-plain-v5.pdf")
                  + r"} in this genome's folder). Record a verdict on it with " + verdict_cmd + r".\par}", ""]
        else:
            cited = {cid for s in summ for cid in (s.get("claims") or [])}
            L += [r"\textbf{Plain-language summary} (" + tex(_status(pl)) + ("" if d.get("summary") else
                  "; offline template") + r"):", "", sentences_tex(summ), ""]
            if d.get("worth_checking"):
                L += [r"\textbf{Worth checking:} " + sentences_tex(d["worth_checking"]), ""]
            L += claims_table([c for c in pl.get("claims") or [] if c["id"] in cited])
            L += ["", r"{\footnotesize Every paragraph and claim: the certificate's ``Plain-language summary'' "
                  r"subsection. Record a verdict on the plain-language layer with " + verdict_cmd + r".\par}", ""]
    L += [r"\textbf{Verdict.}", "",
          r"\begin{tabularx}{\linewidth}{@{}lX@{}}",
          r"Verified (signoff-v1 passes; correct as it stands, or after my changes): & $\square$ yes \quad "
          r"$\square$ no \\[4pt]",
          r"Minutes: & \rule{3cm}{0.4pt} \\[4pt]",
          r"Changes: & \rule{\linewidth}{0.4pt} \\[10pt]",
          r" & \rule{\linewidth}{0.4pt} \\[4pt]",
          r"Notes: & \rule{\linewidth}{0.4pt} \\[10pt]",
          r" & \rule{\linewidth}{0.4pt} \\",
          r"\end{tabularx}", "",
          r"{\raggedright\textbf{Record it} (runs the signoff-v1 check, PREREG amendment 2026-11-11, and is "
          r"refused until it passes; writes \texttt{mike\_verification} into \texttt{"
          + tex(f"{records_dir_shown()}/{lab}") + r".json}; use \texttt{-{}-not-verified} for no; rebuild the CSVs with "
          r"\texttt{uv run python experiments/10seed/build\_explanations\_csv.py}):\par}",
          "", r"{\footnotesize\raggedright\texttt{" + tex(record_command(lab)) + r"}\par}", ""]
    return "\n".join(L) + "\n"


def cover_page(docs: List[Dict[str, Any]], title: str) -> str:
    L = [r"\begin{titlepage}", r"\centering", r"{\Large\bfseries " + tex(title) + r"\par}", r"\vspace{6mm}",
         r"{\small Generated by explaneat export from the JSON records; order as pre-registered "
         r"(\texttt{PRE} in \texttt{experiments/10seed/build\_explanations\_csv.py}).\par}",
         r"\vspace{8mm}", r"\begin{tabular}{@{}rllrrrl@{}}", r"\toprule",
         r"\# & Genome & Dataset & Test AUC & Struct. & Comp. & Recorded verdict \\", r"\midrule"]
    for i, d in enumerate(docs):
        s = d["coverage"]["summary"]
        auc = ((d["model"].get("performance") or {}).get("test") or {}).get("auc")
        hv = d["authorship"]["human_verification"]
        L.append(f"{i + 1} & {tex(d['label'])} & {tex((d['provenance'].get('dataset') or {}).get('name') or 'n/a')} & "
                 f"{tex(fnum(auc, 4))} & {tex(fnum(s['structural_coverage'], 4))} & {tex(fnum(s['compositional_coverage'], 4))} & "
                 + ("verified" if hv and hv.get("verified") else ("not verified" if hv else "none")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\vspace{8mm}",
          r"\par{\small For each genome: a verification checklist (recorded vs live checks, what to look at, "
          r"verdict fields and the command that records the verdict), then its explanation certificate.\par}",
          r"\end{titlepage}", r"\tableofcontents", r"\clearpage"]
    return "\n".join(L) + "\n"


def include_pdf_lines(path: str, rotated: List[int], n_pages: Optional[int], toc_title: str,
                      toc_label: str) -> List[str]:
    """\\includepdf lines for a whole PDF, page for page: runs of portrait pages as they are, pages
    the source shows rotated (landscape figures and tables) on landscape pages, the first page
    carrying the table-of-contents entry and bookmark."""
    if not n_pages:
        return [r"\includepdf[pages=-,addtotoc={1,section,1,{" + toc_title + "}," + toc_label + "}]{" + path + "}"]
    rot = set(rotated)
    runs: List[Tuple[int, int, bool]] = []
    for pg in range(1, n_pages + 1):
        r = pg in rot
        if runs and runs[-1][2] == r and runs[-1][1] == pg - 1:
            runs[-1] = (runs[-1][0], pg, r)
        else:
            runs.append((pg, pg, r))
    out = []
    for i, (a, b, r) in enumerate(runs):
        opts = [f"pages={{{a}-{b}}}" if a != b else f"pages={{{a}}}"]
        if r:
            opts.append("landscape")
        if i == 0:
            opts.append("addtotoc={1,section,1,{" + toc_title + "}," + toc_label + "}")
        out.append(r"\includepdf[" + ",".join(opts) + "]{" + path + "}")
    return out


def render_review(docs: List[Dict[str, Any]], title: str, review_marks: bool = True) -> str:
    """review.tex in the pack root; genome files are under <label>/."""
    L = [r"% Verification review document (generated by explaneat export verification-pack).",
         r"\documentclass[a4paper,11pt]{article}",
         r"\usepackage[margin=2.2cm]{geometry}",
         render_tex.PREAMBLE_PACKAGES.rstrip(),
         r"\usepackage[unicode=true,hidelinks]{hyperref}", r"\setcounter{tocdepth}{1}",
         r"\usepackage{cleveref}", r"\usepackage{pdfpages}", r"\usepackage{longtable}",
         render_tex.DRAFTED_DEFS.rstrip(),
         "" if review_marks else r"\reviewmarksfalse",
         r"\providecommand{\explaneatcertdir}[1]{}",
         r"\renewcommand{\explaneatcertdir}[1]{#1/}",
         r"\begin{document}",
         cover_page(docs, title)]
    for i, d in enumerate(docs):
        L.append(checklist_block(d, i + 1))
        if d.get("plain_pdf"):
            # the complete plain-language (v5 read-down) document, page for page as its standalone
            # PDF: its own figure numbers (1, 2, ...) restart for each genome
            L.append(r"\clearpage")
            L += include_pdf_lines(d["plain_pdf"], d.get("plain_rotated") or [], d.get("plain_pages"),
                                   tex(d["label"]) + r": plain-language explanation (read-down document)",
                                   "plain:" + render_tex.label_key(d["label"]))
            L.append(r"\addcontentsline{toc}{section}{" + tex(d["label"]) + r": coverage certificate}")
            # the read-down document carries the summary, hierarchy, evidence and figures: only the
            # slim coverage certificate follows it
            L.append(f"\\input{{{d['label']}/{d['label']}-coverage.tex}}")
        else:
            L.append(f"\\input{{{d['label']}/{d['label']}.tex}}")
        L.append(r"\clearpage")
    L.append(r"\end{document}")
    return printed("\n".join(L)) + "\n"
