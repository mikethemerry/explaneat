"""The `explaneat demo` hello world: seven stages, each PASS / FAIL / SKIP.

Everything is synthetic and local. Outputs go to ``out_dir`` (default
``./demo_output``, git-ignored); the database, if used, is a SQLite file.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Stage:
    key: str
    title: str
    status: str = SKIP
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoSettings:
    seeds: Sequence[int] = (0, 1)
    n_samples: int = 800
    generations: int = 30
    pop_size: int = 50
    epochs: int = 10
    survival_samples: int = 1000
    survival_generations: int = 25

    @classmethod
    def tiny(cls) -> "DemoSettings":
        return cls(seeds=(0, 1), n_samples=200, generations=2, pop_size=12, epochs=3,
                   survival_samples=300, survival_generations=2)


@dataclass
class DemoResult:
    stages: List[Stage]
    out_dir: Path
    report_md: Optional[Path] = None
    report_html: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return all(s.status != FAIL for s in self.stages)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


class _Demo:
    def __init__(self, out_dir: Path, settings: DemoSettings, database_url: Optional[str],
                 use_db: bool, log: Callable[[str], None]):
        self.out_dir = out_dir
        self.s = settings
        self.requested_db = database_url
        self.use_db = use_db
        self.log = log
        self.db_url: Optional[str] = None
        self.stages: List[Stage] = []
        self.ctx: Dict[str, Any] = {}

    # -- plumbing -----------------------------------------------------------
    def run_stage(self, key: str, title: str, fn: Callable[[Stage], None]) -> Stage:
        st = Stage(key, title)
        self.log(f"[{len(self.stages) + 1}/7] {title} ...")
        t0 = time.time()
        try:
            st.status = PASS
            fn(st)
        except Exception as e:  # a stage failure is reported, not raised
            st.status = FAIL
            st.notes.append(f"{type(e).__name__}: {e}")
            st.metrics["traceback"] = traceback.format_exc(limit=6)
        st.seconds = round(time.time() - t0, 2)
        self.log(f"      {st.status} ({st.seconds}s)" + (f" - {st.notes[0]}" if st.notes else ""))
        self.stages.append(st)
        return st

    # -- 1. database ----------------------------------------------------------
    def stage_database(self, st: Stage) -> None:
        if not self.use_db:
            st.status = SKIP
            st.notes.append("--no-db: training runs in memory")
            return
        from .. import settings
        from ..db.sqlite_support import sqlite_schema_supported

        url = self.requested_db
        if not url:
            configured = settings.read_config().get("database_url", "")
            url = configured if settings.is_sqlite(configured) else settings.sqlite_url(
                self.out_dir / "demo.db")
        st.metrics["database"] = settings.redact(url)
        if settings.is_sqlite(url):
            ok, how = sqlite_schema_supported()
            if not ok:
                st.status = SKIP
                st.notes.append(how)
                st.notes.append("training runs in memory; ancestry is not recorded")
                return
            st.notes.append(f"SQLite schema support: {how}")
        from ..db.base import db
        from ..db.models import Experiment

        db.init_db(url)
        db.create_all()
        with db.session_scope() as session:
            st.metrics["experiments_before"] = session.query(Experiment).count()
        self.db_url = url

    # -- 2. data --------------------------------------------------------------
    def stage_data(self, st: Stage) -> None:
        from .data import hinge_classification, train_test_split_indices

        X, y, truth = hinge_classification(self.s.n_samples, 4, seed=0)
        tr, te = train_test_split_indices(len(X), 0.3, seed=0)
        self.ctx["clf"] = dict(X=X, y=y, tr=tr, te=te, truth=truth)
        st.metrics["classification"] = {"n": len(X), "features": X.shape[1], "train": len(tr),
                                        "test": len(te), "positive_rate": round(float(y.mean()), 3),
                                        "truth": truth["logit"]}
        if self.db_url:
            from ..db.dataset_utils import save_dataset_split_with_indices, save_dataset_to_db

            ds = save_dataset_to_db(
                "demo-hinge (synthetic)", X, y, source="synthetic",
                description=f"explaneat demo: {truth['feature']}",
                feature_names=[f"x{i}" for i in range(X.shape[1])], target_name="y",
                class_names=["0", "1"], metadata={"generator": "explaneat.demo.data"})
            from sqlalchemy import inspect as sa_inspect

            ds_id = sa_inspect(ds).identity[0]  # the returned row is detached
            split = save_dataset_split_with_indices(ds_id, tr.tolist(), te.tolist(),
                                                    test_size=0.3, random_state=0)
            split_id = sa_inspect(split).identity[0]
            self.ctx["dataset_id"], self.ctx["split_id"] = str(ds_id), str(split_id)
            st.notes.append("classification dataset and split saved to the database")

        if _module_available("explaneat.survival.simulate"):
            from ..survival.simulate import simulate

            Xs, t, e, g = simulate(n=self.s.survival_samples, d=4, mode="nonlinear", seed=0)
            str_, ste = train_test_split_indices(len(Xs), 0.3, seed=1)
            self.ctx["surv"] = dict(X=Xs, time=t, event=e, g=g, tr=str_, te=ste)
            st.metrics["survival"] = {"n": len(Xs), "features": Xs.shape[1],
                                      "event_rate": round(float(e.mean()), 3),
                                      "truth": "log(5) * exp(-(x0^2 + x1^2) / 0.5), zero linear effect"}
        else:
            st.notes.append("survival data skipped: explaneat.survival is not on this branch "
                            "(arrives with the exp/cox-survival merge)")

    # -- 3. training ------------------------------------------------------------
    def stage_train(self, st: Stage) -> None:
        from .training import train_classifier

        c = self.ctx["clf"]
        runs = []
        for seed in self.s.seeds:
            t0 = time.time()
            genome, config, info = train_classifier(
                c["X"][c["tr"]], c["y"][c["tr"]], seed=seed, generations=self.s.generations,
                pop_size=self.s.pop_size, epochs=self.s.epochs, work_dir=self.out_dir,
                database_url=self.db_url, experiment_name=f"demo-hinge-s{seed}")
            info["seconds"] = round(time.time() - t0, 1)
            if self.db_url:
                self._link_split(info["experiment_id"], seed)
            runs.append(dict(genome=genome, config=config, info=info))
            st.metrics[f"classification_s{seed}"] = info
        self.ctx["clf_runs"] = runs

        if "surv" in self.ctx:
            self.ctx["surv_runs"] = self._train_survival(st)

    def _link_split(self, experiment_id: str, seed: int) -> None:
        from uuid import UUID

        from ..db.base import db
        from ..db.models import Experiment

        with db.session_scope() as session:
            exp = session.get(Experiment, UUID(experiment_id))
            if exp is not None:
                exp.dataset_id = UUID(self.ctx["dataset_id"])
                exp.split_id = UUID(self.ctx["split_id"])
                exp.random_seed = seed

    def _train_survival(self, st: Stage) -> List[dict]:
        import copy

        from ..core.backproppop import BackpropPopulation
        from ..core.neuralneat import set_output_activation
        from ..survival import CoxCriterion, make_cox_fitness
        from .training import CLASSIFICATION_CONFIG, load_config, seed_everything

        sv = self.ctx["surv"]
        Xtr = sv["X"][sv["tr"]]
        ytr = np.column_stack([sv["time"][sv["tr"]], sv["event"][sv["tr"]].astype(float)])
        runs = []
        for seed in self.s.seeds:
            t0 = time.time()
            text = CLASSIFICATION_CONFIG.format(pop_size=self.s.pop_size, num_inputs=Xtr.shape[1])
            text = text.replace("num_hidden = 2", "num_hidden = 1").replace(
                "weight_init_stdev = 1.0", "weight_init_stdev = 0.1")
            config = load_config(text, self.out_dir)
            set_output_activation(config, "identity")
            seed_everything(seed)
            pop = BackpropPopulation(config, Xtr, ytr, criterion=CoxCriterion("efron"),
                                     mask_phantom=True, skip_invalid_genomes=True)
            pop.run(make_cox_fitness("efron", 1.0), n=self.s.survival_generations,
                    nEpochs=self.s.epochs)
            info = {"seed": seed, "generations": self.s.survival_generations,
                    "pop_size": self.s.pop_size, "mode": "in-memory",
                    "train_fitness": float(pop.best_genome.fitness),
                    "seconds": round(time.time() - t0, 1)}
            runs.append(dict(genome=copy.deepcopy(pop.best_genome), config=config, info=info))
            st.metrics[f"survival_s{seed}"] = info
        return runs

    # -- 4. explanation ---------------------------------------------------------
    def stage_explain(self, st: Stage) -> None:
        from .explain import explain_genome

        problems = []
        for task, key in (("classification", "clf"), ("survival", "surv")):
            if f"{key}_runs" not in self.ctx:
                continue
            X = self.ctx[key]["X"][self.ctx[key]["te"]]
            for run in self.ctx[f"{key}_runs"]:
                r = explain_genome(run["genome"], run["config"], X)
                run["explanation"] = r
                seed = run["info"]["seed"]
                st.metrics[f"{task}_s{seed}"] = r.summary()
                if r.structural_coverage < 1.0 or r.compositional_coverage < 1.0:
                    problems.append(f"{task} s{seed}: coverage {r.structural_coverage:.3f} / "
                                    f"{r.compositional_coverage:.3f}")
                worst = max(r.max_abs_diff.values())
                if not worst < 1e-9:
                    problems.append(f"{task} s{seed}: function not preserved (max diff {worst:.2e})")
                if self.db_url and task == "classification":
                    st.notes.append(self._store_explanation(run))
        if problems:
            raise AssertionError("; ".join(problems))
        st.notes.append("every operation validated; coverage 1.0/1.0; trained network = "
                        "phenotype = explained model = root annotation function")

    def _store_explanation(self, run: dict) -> str:
        """Save the operations on the best genome's DB row and replay them."""
        from uuid import UUID

        from ..core.model_state import ModelStateEngine
        from ..db.base import db
        from ..db.models import Explanation, Genome, Population

        r = run["explanation"]
        with db.session_scope() as session:
            row = (session.query(Genome).join(Population, Genome.population_id == Population.id)
                   .filter(Population.experiment_id == UUID(run["info"]["experiment_id"]),
                           Genome.genome_id == run["genome"].key)
                   .order_by(Population.generation.desc()).first())
            if row is None:
                return "best genome row not found; explanation not stored"
            session.add(Explanation(genome_id=row.id, name="demo scripted explanation",
                                    description="explaneat demo: splits + leaf/composition annotations",
                                    is_well_formed=True,
                                    structural_coverage=r.structural_coverage,
                                    compositional_coverage=r.compositional_coverage,
                                    operations=r.operations))
            genome_row_id = str(row.id)
        # Round trip: rebuild the phenotype from the stored genome, replay.
        from mcp_server.helpers import build_engine

        with db.session_scope() as session:
            engine = build_engine(session, genome_row_id)
            n = len(engine.annotations)
        if n != len(r.leaves) + len(r.compositions):
            raise AssertionError(f"stored explanation replays {n} annotations, expected "
                                 f"{len(r.leaves) + len(r.compositions)}")
        run["info"]["genome_row_id"] = genome_row_id
        return f"seed {run['info']['seed']}: explanation stored on genome {genome_row_id} and replayed"

    # -- 5. evidence and metrics --------------------------------------------------
    def stage_metrics(self, st: Stage) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        from ..analysis.viz_data import compute_sensitivity
        from .training import predict

        c = self.ctx["clf"]
        Xtr, ytr, Xte, yte = c["X"][c["tr"]], c["y"][c["tr"]], c["X"][c["te"]], c["y"][c["te"]]
        logit_true = 4.0 * (np.abs(Xte[:, 0]) - 0.5) + 1.5 * Xte[:, 1]
        lr = LogisticRegression().fit(Xtr, ytr)
        st.metrics["classification_reference"] = {
            "oracle_auc": round(float(roc_auc_score(yte, logit_true)), 4),
            "logistic_regression_auc": round(float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])), 4),
        }
        bad = []
        for run in self.ctx["clf_runs"]:
            seed = run["info"]["seed"]
            auc_te = float(roc_auc_score(yte, predict(run["genome"], run["config"], Xte)))
            auc_tr = float(roc_auc_score(ytr, predict(run["genome"], run["config"], Xtr)))
            m = {"train_auc": round(auc_tr, 4), "test_auc": round(auc_te, 4)}
            r = run.get("explanation")
            if r is not None and r.engine is not None:
                sens = self._root_sensitivity(r, Xte, compute_sensitivity)
                m["root_sensitivity"] = sens
            st.metrics[f"classification_s{seed}"] = m
            if not (_finite(auc_te) and auc_te > 0.5):
                bad.append(f"classification s{seed}: test AUC {auc_te:.3f} is not above chance")
        if "surv_runs" in self.ctx:
            self._survival_metrics(st, bad)
        else:
            st.notes.append("survival metrics skipped (explaneat.survival not on this branch)")
        if bad:
            raise AssertionError("; ".join(bad))

    @staticmethod
    def _root_sensitivity(r, X, compute_sensitivity) -> Dict[str, float]:
        from ..analysis.annotation_function import AnnotationFunction
        from .explain import _input_column

        anns = {a.name: a for a in r.engine.annotations}
        root = anns[r.root]
        state = r.engine.current_state
        fn = AnnotationFunction.from_structure(
            {"entry_nodes": list(root.entry_nodes), "exit_nodes": list(root.exit_nodes),
             "subgraph_nodes": sorted({str(n) for a in anns.values() for n in a.subgraph_nodes}),
             "subgraph_connections": [list(map(str, e)) for a in anns.values()
                                      for e in a.subgraph_connections]}, state)
        cols = [_input_column(e) for e in root.entry_nodes]
        sens = compute_sensitivity(fn, X[:, cols], entry_names=list(root.entry_nodes))
        per_feature: Dict[str, float] = {}
        for name, score in zip(sens["input_labels"], next(iter(sens["sensitivities"].values()))):
            feat = f"x{_input_column(name)}"
            per_feature[feat] = round(per_feature.get(feat, 0.0) + score, 4)
        return dict(sorted(per_feature.items()))

    def _survival_metrics(self, st: Stage, bad: List[str]) -> None:
        from ..survival import BreslowBaseline
        from ..survival.metrics import harrell_c, survival_metrics
        from .training import predict

        sv = self.ctx["surv"]
        tr, te = sv["tr"], sv["te"]
        t_tr, e_tr, t_te, e_te = sv["time"][tr], sv["event"][tr], sv["time"][te], sv["event"][te]
        horizons = [float(q) for q in np.quantile(t_tr[e_tr], [0.25, 0.5, 0.75])]
        st.metrics["survival_reference"] = {"oracle_harrell_c": round(harrell_c(sv["g"][te], t_te, e_te), 4)}
        for run in self.ctx["surv_runs"]:
            seed = run["info"]["seed"]
            g_tr = predict(run["genome"], run["config"], sv["X"][tr])
            g_te = predict(run["genome"], run["config"], sv["X"][te])
            bl = BreslowBaseline().fit(g_tr, t_tr, e_tr)
            m = survival_metrics(g_tr, g_te, t_tr, e_tr, t_te, e_te, baseline=bl,
                                 horizons=horizons, calibration_horizon=horizons[1])
            keep = {"harrell_c": m.get("harrell_c"), "uno_c": m.get("uno_c"),
                    "td_auc_mean": (m.get("td_auc") or {}).get("mean_auc"),
                    "ibs": m.get("ibs")}
            cal = m.get("calibration")
            if isinstance(cal, dict):
                keep["calibration"] = {k: v for k, v in cal.items()
                                       if isinstance(v, (int, float, str)) and k != "bins"}
            st.metrics[f"survival_s{seed}"] = json.loads(json.dumps(keep, default=float))
            if not (_finite(keep["harrell_c"]) and keep["harrell_c"] > 0.5):
                bad.append(f"survival s{seed}: Harrell C {keep['harrell_c']} is not above chance")

    # -- 6. lineage -------------------------------------------------------------
    def stage_lineage(self, st: Stage) -> None:
        if self.db_url:
            from uuid import UUID

            from ..db.base import db
            from ..db.models import Genome, Population

            with db.session_scope() as session:
                for run in self.ctx.get("clf_runs", []):
                    exp_id = UUID(run["info"]["experiment_id"])
                    q = (session.query(Genome).join(Population, Genome.population_id == Population.id)
                         .filter(Population.experiment_id == exp_id))
                    st.metrics[f"classification_s{run['info']['seed']}"] = {
                        "genomes_recorded": q.count(),
                        "with_parents": q.filter(Genome.parent1_id.isnot(None)).count()}
        if not _module_available("explaneat.analysis.lineage"):
            st.status = SKIP
            st.notes.append("explaneat/analysis/lineage.py is not on this branch; "
                            "lineage computation skipped" +
                            ("" if self.db_url else " (and no database, so no ancestry rows)"))
            return
        st.status = SKIP
        st.notes.append("explaneat.analysis.lineage is present but the demo has no hook for it yet")

    # -- 7. report ----------------------------------------------------------------
    def stage_report(self, st: Stage) -> None:
        from .report import write_reports

        md, html = write_reports(self.out_dir, self.stages + [st], self.environment())
        self.ctx["report"] = (md, html)
        st.metrics["files"] = [md.name, html.name]

    def environment(self) -> Dict[str, Any]:
        import torch

        from .. import __version__

        return {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "platform": platform.platform(),
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "explaneat": __version__,
                "settings": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in asdict(self.s).items()},
                "database": self._db_label()}

    def _db_label(self) -> Optional[str]:
        if not self.db_url:
            return None
        from .. import settings

        return settings.redact(self.db_url)


def run_demo(out_dir: os.PathLike = "demo_output", *, tiny: bool = False,
             settings: Optional[DemoSettings] = None, database_url: Optional[str] = None,
             use_db: bool = True, log: Callable[[str], None] = print) -> DemoResult:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    demo = _Demo(out, settings or (DemoSettings.tiny() if tiny else DemoSettings()),
                 database_url, use_db, log)
    demo.run_stage("database", "Database (SQLite)", demo.stage_database)
    data = demo.run_stage("data", "Synthetic data", demo.stage_data)
    deps_ok = data.status == PASS
    for key, title, fn in (("train", "Training (2 seeds per task)", demo.stage_train),
                           ("explain", "Scripted explanation", demo.stage_explain),
                           ("metrics", "Evidence and metrics", demo.stage_metrics)):
        if deps_ok:
            deps_ok = demo.run_stage(key, title, fn).status == PASS
        else:
            demo.stages.append(Stage(key, title, FAIL, notes=["not run: an earlier stage failed"]))
    if deps_ok:
        demo.run_stage("lineage", "Lineage", demo.stage_lineage)
    else:
        demo.stages.append(Stage("lineage", "Lineage", SKIP, notes=["not run: an earlier stage failed"]))
    demo.run_stage("report", "Report", demo.stage_report)
    if demo.db_url:
        from ..db.base import db

        db.close()
    md, html = demo.ctx.get("report", (None, None))
    return DemoResult(demo.stages, out, md, html)
