"""Rate--distortion instrumentation for the BeamStar filtering block.

Implements the per-step ledger of Alg. 1 / Remark (two decompositions):
    Delta_D_s = -log( retained mass fraction at step s )   [total nats]
which by the chain rule telescopes EXACTLY to
    D_tot = -log P(S) = KL(Q_theta || P)
(the closed form of Lemma 1). All quantities are computed from cumulative
logprobs already in hand -- no extra forward passes.

Attribution note: Delta_D is measured against the *post-top-k pool*
(`top_beams_logprob` BEFORE BeamStar filtering), so the ledger charges
BeamStar only for mass discarded beyond the same-width baseline's own
truncation -- the correct baseline for the paper's comparisons.

Temperature note: if logprobs are temperature-scaled, the ledger measures
distortion w.r.t. the scaled model; use raw logprobs to measure w.r.t. p_phi.
"""

import math
import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp of a 1-D array of logprobs."""
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


class RDLedger:
    """Per-request rate--distortion ledger (Alg. 1, lines 5-8)."""

    def __init__(self, delta_budget_nats: float = float("inf")):
        # Set e.g. delta_budget_nats = 0.05 * expected_len for a per-symbol
        # budget of delta = 0.05 nats/token; inf disables the guard
        # (measurement-only mode).
        self.delta_budget = delta_budget_nats
        self.D_tot = 0.0          # accumulated distortion, total nats
        self.delta_D = []         # per-step increments (nats)
        self.kept = []            # per-step active beam count  -> rate R(theta)
        self.pool = []            # per-step pool size (post top-k baseline)
        self.eps_realized = []    # realized margin, rate units (schedule diagnostics)

    # ------------------------------------------------------------------ #
    def step(self, top_beams_logprob: np.ndarray, keep_pos: np.ndarray,
             token: int):
        """Charge one decoding step. Returns (possibly grown) keep_pos.

        top_beams_logprob : cumulative logprobs of the post-top-k pool
        keep_pos          : indices into top_beams_logprob kept by the cap
        token             : 0-based step index (so t = token + 1)
        """
        lse_pool = _logsumexp(top_beams_logprob)
        lse_kept = _logsumexp(top_beams_logprob[keep_pos])
        dD = lse_pool - lse_kept                     # >= 0, Lemma 1 incremental form

        # --- budget guard (Alg. 1, line 7): re-admit greedily ----------- #
        if math.isfinite(self.delta_budget) and self.D_tot + dD > self.delta_budget:
            pruned = np.setdiff1d(
                np.arange(top_beams_logprob.size, dtype=np.intp), keep_pos)
            # re-admit in decreasing probability (Prop. greedy-opt: this is
            # the distortion-optimal order for a given cardinality)
            for j in pruned[np.argsort(-top_beams_logprob[pruned])]:
                keep_pos = np.append(keep_pos, j)
                lse_kept = _logsumexp(top_beams_logprob[keep_pos])
                dD = lse_pool - lse_kept
                if self.D_tot + dD <= self.delta_budget:
                    break
            keep_pos = np.sort(keep_pos)

        # --- ledger ------------------------------------------------------ #
        self.D_tot += dD
        self.delta_D.append(dD)
        self.kept.append(int(keep_pos.size))
        self.pool.append(int(top_beams_logprob.size))
        # realized margin in rate units: (r_worst_kept - r_best) at this step;
        # compare against the theorem schedule eps_t ~ eps/2 + sigma*sqrt(2 log(T/d)/t)
        t = token + 1
        r = -top_beams_logprob[keep_pos] / t
        self.eps_realized.append(float(r.max() - r.min()))
        return keep_pos

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        T = max(len(self.kept), 1)
        return {
            "D_tot_nats": self.D_tot,                        # KL(Q||P), Lemma 1
            "D_per_symbol": self.D_tot / T,                  # per-symbol form
            "retained_mass": math.exp(-self.D_tot),          # P(S)
            "recall_risk_bound": 1.0 - math.exp(-self.D_tot),# Prop. recall, c=1
            "R_operational": float(np.mean(self.kept)),      # E|B_t|
            "R_info_bits": float(np.mean(np.log2(np.maximum(self.kept, 1)))),
            "delta_D": self.delta_D,                         # chain-rule ledger
            "kept": self.kept,
            "eps_realized": self.eps_realized,
        }


# ---------------------------------------------------------------------- #
# Integration into the user's filtering block (patched vLLM beam_search):
# ---------------------------------------------------------------------- #
#
#   # per request, before the decoding loop:
#   ledger = RDLedger(delta_budget_nats=float("inf"))   # or a finite budget
#
#   BEAM_STAR_FILTERING = True
#   if BEAM_STAR_FILTERING:
#       learn_threshold = True
#       if cum_threshold is not None or learn_threshold and token < (max_tokens - 1):
#           top_beams_logprob = all_beams_logprob[topn_idx]
#           if learn_threshold:
#               self.req_stats[token].push(top_beams_logprob)
#               stats = self.req_stats[token]
#               thr = stats.max_val - 2 * (token + 1) * stats.std
#           else:
#               thr = self.thr[token]
#
#           keep_pos = np.flatnonzero(top_beams_logprob >= thr)
#           if keep_pos.size == 0:
#               keep_pos = np.array([0], dtype=np.intp)   # >= 1 active beam
#
#           # >>> the one new line (plus guard/re-admission inside):
#           keep_pos = ledger.step(top_beams_logprob, keep_pos, token)
#
#           print(f"token {token}: kept {keep_pos.size}  "
#                 f"dD {ledger.delta_D[-1]:.4f}  D_tot {ledger.D_tot:.4f}  "
#                 f"mass {math.exp(-ledger.D_tot):.4f}")
#           topn_idx = topn_idx[keep_pos]
#
#   # at end of generation, attach to the request output / metrics sink:
#   rd = ledger.summary()