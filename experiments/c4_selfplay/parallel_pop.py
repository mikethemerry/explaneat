"""Population backprop in a pool of worker processes, bitwise identical to the serial loop.

``ParallelBackpropPopulation(config, xs, ys, ..., workers=N)`` is a ``BackpropPopulation`` whose
``backpropagate`` trains the genomes in ``N`` spawned worker processes instead of one after
another. Everything else (``run``, reproduction, speciation, reporters) is the parent's code.

**Why it is identical.** The serial per-genome step (``BackpropPopulation.backpropagate``) is a
pure function of (genome, config, criterion, mask_phantom, weight_decay, nEpochs, xs, ys): it
builds ``NeuralNeat``, runs full-batch Adadelta(lr=1.5) for ``nEpochs`` and writes the trained
weights/biases back into the *same* genome object (``self.population[k] = net.genome`` is the
object already there). It draws no random numbers (no python ``random``, numpy or torch RNG:
NeuralNeat construction, forward/backward and Adadelta are deterministic), so genomes can be
trained in any process in any order. ``_train_one`` below repeats that step line for line in a
worker with the same torch settings (CPU, float64, ``torch.set_num_threads(1)``,
``use_deterministic_algorithms(True)``) and returns the written-back values (python floats, which
pickle exactly) and the per-genome stats. The main process then walks the population in its own
order, writes the values into the existing genome objects (so species members, which reference
those objects, see them) and accumulates the stats exactly as the serial loop does, so
``generation_details``, ``backprop_times`` and the log lines come out in the same order.

Bitwise equality holds against a serial run with ``torch.set_num_threads(1)`` (``run.py`` uses
1): BLAS reductions over the rows can change with the thread count.

**Data.** ``xs``/``ys`` and a pickle of (config, criterion, mask_phantom, weight_decay) are
written once per change to a private temp folder (``.npy`` + ``.pkl``); each task carries only
that folder's path, the genome and nEpochs, and a worker reloads when the path changes. The
round loop replaces ``pop.xs``/``pop.ys`` each round: a new tensor (or an in-place edit, seen
through ``tensor._version``) or a changed context gives a new folder, so the pool itself
persists across generations and rounds. It is created lazily on the first call and closed by
``close()``, the context manager or ``atexit``. ``workers <= 1`` uses the serial parent method.

**Invalid genomes.** A genome NeuralNeat rejects is skipped with ``skip_invalid_genomes``; without
it the serial loop raises ``GenomeNotValidError`` at that genome after training the earlier ones,
and so does this one (the earlier genomes' results are applied first, later ones discarded).
"""
import atexit
import os
import pickle
import shutil
import tempfile
import time
import weakref

os.environ.setdefault("EXPLANEAT_DEVICE", "cpu")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from neat.math_util import mean  # noqa: E402

from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

# ---------------------------------------------------------------- worker side

_W = {"dir": None, "xs": None, "ys": None, "ctx": None}


def _worker_init():
    os.environ["EXPLANEAT_DEVICE"] = "cpu"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _load_state(state_dir):
    if _W["dir"] == state_dir:
        return
    _W["xs"] = torch.from_numpy(np.load(os.path.join(state_dir, "xs.npy")))
    _W["ys"] = torch.from_numpy(np.load(os.path.join(state_dir, "ys.npy")))
    with open(os.path.join(state_dir, "ctx.pkl"), "rb") as f:
        _W["ctx"] = pickle.load(f)
    _W["dir"] = state_dir


def _train_one(task):
    """One genome's serial backprop step. Returns ``(key, None)`` for an invalid genome, else
    ``(key, (conn_weights, node_biases, stats))`` with exactly the values
    ``NeuralNeat.update_genome_weights`` writes."""
    state_dir, key, genome, nEpochs = task
    _load_state(state_dir)
    xs, ys = _W["xs"], _W["ys"]
    ctx = _W["ctx"]
    try:
        net = NeuralNeat(genome, ctx["config"], criterion=ctx["criterion"],
                         mask_phantom=ctx["mask_phantom"])
    except GenomeNotValidError:
        return key, None

    optimizer = optim.Adadelta(net.parameters(), lr=1.5, weight_decay=ctx["weight_decay"])
    optimizer.zero_grad()
    bce = ctx["criterion"]
    loss_preds = net.forward(xs)
    preBPLoss = bce(loss_preds, ys)

    start_time = time.time()
    for _ in range(nEpochs):
        preds = net.forward(xs)
        loss = bce(preds, ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    end_time = time.time()
    avg_time_per_epoch = (end_time - start_time) / nEpochs

    postBPLoss = bce(net.forward(xs), ys)
    lossDiff = postBPLoss - preBPLoss

    net.update_genome_weights()
    # exactly the entries update_genome_weights wrote
    weights, biases = {}, {}
    for layer in net.layers.values():
        for genome_location in layer["input_map"]:
            weights[genome_location] = net.genome.connections[genome_location].weight
        for node_id in layer["nodes"]:
            if node_id in net.genome.nodes:
                biases[node_id] = net.genome.nodes[node_id].bias
    stats = dict(size=len(net.genome.nodes), depth=net.node_mapping.depth,
                 width=net.node_mapping.width, avg_time_per_epoch=avg_time_per_epoch,
                 improvement=lossDiff.item(), post_loss=postBPLoss.item())
    return key, (weights, biases, stats)


# ---------------------------------------------------------------- main side

_LIVE = weakref.WeakSet()


@atexit.register
def _close_all():
    for pop in list(_LIVE):
        pop.close()


class ParallelBackpropPopulation(BackpropPopulation):
    """``BackpropPopulation`` with ``backpropagate`` run in ``workers`` processes."""

    def __init__(self, *args, workers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = int(workers)
        self._pool = None
        self._tmp = None
        self._state_dir = None
        self._state_key = None
        self._state_refs = None
        self._ctx_bytes = None
        self._n_states = 0
        _LIVE.add(self)

    # -- lifecycle
    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
        self._state_dir = self._state_key = self._state_refs = self._ctx_bytes = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __getstate__(self):  # the pool and temp folder are process-local
        state = self.__dict__.copy()
        for k in ("_pool", "_tmp", "_state_dir", "_state_key", "_state_refs", "_ctx_bytes"):
            state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        _LIVE.add(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init)
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="c4_parallel_pop_")

    def _ensure_state(self, xs, ys):
        """Write xs/ys/context for the workers when any of them changed; return the folder."""
        ctx_bytes = pickle.dumps(dict(config=self.config, criterion=self.criterion,
                                      mask_phantom=self.mask_phantom,
                                      weight_decay=self.weight_decay))
        key = (id(xs), xs._version, id(ys), ys._version)
        if (self._state_dir is not None and key == self._state_key
                and ctx_bytes == self._ctx_bytes):
            return self._state_dir
        old = self._state_dir
        self._n_states += 1
        d = os.path.join(self._tmp, "s%d" % self._n_states)
        os.makedirs(d)
        np.save(os.path.join(d, "xs.npy"), xs.detach().cpu().numpy())
        np.save(os.path.join(d, "ys.npy"), ys.detach().cpu().numpy())
        with open(os.path.join(d, "ctx.pkl"), "wb") as f:
            f.write(ctx_bytes)
        self._state_dir, self._state_key, self._ctx_bytes = d, key, ctx_bytes
        self._state_refs = (xs, ys)  # keep alive so the ids are not reused
        if old is not None:  # every task of the previous call has finished
            shutil.rmtree(old, ignore_errors=True)
        return d

    # -- the step
    def backpropagate(self, xs, ys, nEpochs=5):
        if self.workers <= 1:
            return super().backpropagate(xs, ys, nEpochs=nEpochs)
        self.logger.info("about to start backprop with {} epochs".format(nEpochs))
        try:
            nEpochs = self.config.generations_of_backprop
        except AttributeError:
            nEpochs = nEpochs
        improvements = []
        postLosses = []
        avg_times_per_epoch = []
        size_per_genome = []
        depth_per_genome = []
        width_per_genome = []

        self._ensure_pool()
        state_dir = self._ensure_state(xs, ys)
        items = list(self.population.items())
        tasks = [(state_dir, k, g, nEpochs) for k, g in items]
        results = self._pool.imap(_train_one, tasks, chunksize=1)

        for (k, genome), (rk, res) in zip(items, results):
            assert rk == k
            if res is None:
                if not self.skip_invalid_genomes:
                    for _ in results:  # drain so the pool is idle for the next call
                        pass
                    raise GenomeNotValidError("No valid nodes found in genome")
                continue
            weights, biases, st = res
            for ck, w in weights.items():
                genome.connections[ck].weight = w
            for nk, b in biases.items():
                genome.nodes[nk].bias = b

            avg_times_per_epoch.append(st["avg_time_per_epoch"])
            size_per_genome.append(st["size"])
            depth_per_genome.append(st["depth"])
            width_per_genome.append(st["width"])
            self.backprop_times.append(st["avg_time_per_epoch"])
            improvements.append(st["improvement"])
            self.population[k] = genome
            postLosses.append(st["post_loss"])

        generation_detail = {
            "mean_size_per_genome": mean(size_per_genome),
            "mean_depth_per_genome": mean(depth_per_genome),
            "mean_width_per_genome": mean(width_per_genome),
            "mean_time_per_epoch": mean(avg_times_per_epoch),
            "total_time_per_backprop": sum(avg_times_per_epoch) * nEpochs,
            "mean_improvement": mean(improvements),
            "best_improvement": min(improvements),
            "best_loss": min(postLosses),
        }
        self.generation_details.append(generation_detail)

        self.logger.info("mean improvement: %s" % mean(improvements))
        self.logger.info("best improvement: %s" % min(improvements))
        self.logger.info("best loss: %s" % min(postLosses))
        self.logger.info(
            f"Average time per epoch: {mean(avg_times_per_epoch):.4f} seconds"
        )
        self.logger.info(
            f"Total time per backprop: {sum(avg_times_per_epoch)*nEpochs} seconds"
        )
