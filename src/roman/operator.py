from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _mu_k(num_channels: int) -> float:
    """
    Return the expected number of mixed channels per sampled kernel.

    The expectation follows the same sampling rule used in the ROMAN paper code:

    - `m = min(num_channels, 9)`
    - `U ~ Uniform(0, log2(m + 1))`
    - `K = floor(2 ** U)`

    Under that scheme:

        E[K] = m - ln(m!) / ln(m + 1)

    Parameters
    ----------
    num_channels : int
        Number of pseudochannels available to the downstream ROCKET-style model.

    Returns
    -------
    float
        Expected number of mixed channels per sampled kernel.
    """
    if num_channels <= 0:
        raise ValueError("num_channels must be >= 1")

    m = min(num_channels, 9)
    ln_fact = math.lgamma(m + 1.0)
    ln_mp1 = math.log(m + 1.0)
    return m - (ln_fact / ln_mp1)


def choose_S_roman(
    *,
    C: int,
    alpha: float,
    L: int,
    min_timesteps_per_channel: int,
    S_exact: Optional[int] = None,
    max_pseudochannels: Optional[int] = None,
    N: Optional[int] = None,
    H: Optional[float] = None,
    window_rule: str = "overlap",
    window_surplus: int = 0,
    S_max: Optional[int] = None,
) -> Tuple[int, List[int], List[int], int]:
    """
    Choose the number of ROMAN scales under one of three selection modes.

    ROMAN can be configured in one of three mutually exclusive ways:

    - exact-scale mode: provide `S_exact`
    - pseudochannel-budget mode: provide `max_pseudochannels`
    - coverage mode: provide both `N` and `H`

    Two constraints govern the final selection:

    1. Mobility constraint
       The coarsest representation must still have at least
       `min_timesteps_per_channel` time steps.

    2. Capacity constraint
       Depending on the chosen mode, the selected scale count must either:
       use exactly `S_exact` when feasible, stay below the pseudochannel budget,
       or satisfy the expected coverage target used in the paper.

    Parameters
    ----------
    C : int
        Number of original input channels.
    alpha : float
        Overlap fraction in `[0, 1)`. Used when `window_rule="overlap"`.
    L : int
        Original time-series length.
    min_timesteps_per_channel : int
        Minimum allowed length for the coarsest scale.
    S_exact : int, optional
        Exact number of scales to request.
    max_pseudochannels : int, optional
        Maximum number of pseudochannels allowed after ROMAN.
    N : int, optional
        Number of channel-combination draws used by the downstream ROCKET-style
        transform. Required together with `H` in coverage mode.
    H : float, optional
        Minimum expected coverage per pseudochannel. Required together with `N`.
    window_rule : {"overlap", "surplus"}, default="overlap"
        Rule used to decide how many windows are extracted at each scale.
    window_surplus : int, default=0
        Extra number of windows added on top of the minimum coverage when using
        `window_rule="surplus"`.
    S_max : int, optional
        Optional hard cap on the explored number of scales.

    Returns
    -------
    tuple
        `(S_star, lengths, windows, L_base)` where:

        - `S_star` is the selected number of scales
        - `lengths` contains one sequence length per scale
        - `windows` contains one window count per scale
        - `L_base` is the coarsest-scale length used as the common output window
          length
    """
    if C <= 0:
        raise ValueError("C must be >= 1")
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1)")
    if L <= 0:
        raise ValueError("L must be >= 1")
    if min_timesteps_per_channel <= 0:
        raise ValueError("min_timesteps_per_channel must be >= 1")
    if window_rule not in {"overlap", "surplus"}:
        raise ValueError("window_rule must be 'overlap' or 'surplus'")
    if window_surplus < 0:
        raise ValueError("window_surplus must be >= 0")
    if S_max is not None and S_max < 1:
        raise ValueError("S_max must be >= 1")

    using_exact = S_exact is not None
    using_budget = max_pseudochannels is not None
    using_coverage = (N is not None) or (H is not None)

    mode_count = sum([using_exact, using_budget, using_coverage])
    if mode_count != 1:
        raise ValueError(
            "Provide exactly one of S_exact, max_pseudochannels, or the pair (N, H)."
        )

    if using_exact:
        if S_exact is None or S_exact <= 0:
            raise ValueError("S_exact must be an integer >= 1.")
    elif using_budget:
        if max_pseudochannels is None or max_pseudochannels <= 0:
            raise ValueError("max_pseudochannels must be an integer >= 1.")
    else:
        if N is None or H is None:
            raise ValueError("Coverage mode requires both N and H.")
        if N <= 0:
            raise ValueError("N must be >= 1.")
        if H <= 0:
            raise ValueError("H must be > 0.")

    def next_len(length_s: int) -> int:
        if length_s <= 1:
            return 1
        if (length_s % 2) == 1:
            length_s += 1
        return max(1, length_s // 2)

    lengths: List[int] = [L]
    while lengths[-1] > 1:
        if S_max is not None and len(lengths) >= S_max:
            break
        lengths.append(next_len(lengths[-1]))

    if L < min_timesteps_per_channel:
        warnings.warn(
            (
                f"Initial length L={L} is smaller than "
                f"min_timesteps_per_channel={min_timesteps_per_channel}. "
                "Falling back to S=1."
            ),
            stacklevel=2,
        )
        W_1 = 1 if window_rule == "overlap" else (1 + window_surplus)
        return 1, [L], [W_1], L

    def compute_windows_for_S(num_scales: int, L_base: int) -> List[int]:
        windows: List[int] = []
        for s in range(1, num_scales + 1):
            Ls = lengths[s - 1]
            if window_rule == "overlap":
                if Ls <= L_base:
                    windows.append(1)
                else:
                    denom = (1.0 - alpha) * float(L_base)
                    extra = int(math.ceil((Ls - L_base) / max(1e-12, denom)))
                    windows.append(1 + extra)
            else:
                min_windows = int(math.ceil(Ls / float(L_base)))
                windows.append(min_windows + window_surplus)
        return windows

    best_S = 1
    best_lengths: List[int] = [L]
    best_windows: List[int] = [1 if window_rule == "overlap" else (1 + window_surplus)]
    best_L_base = L
    constraint_met = False

    for S in range(1, len(lengths) + 1):
        L_base = lengths[S - 1]
        if L_base < min_timesteps_per_channel:
            break

        if using_exact:
            if S > S_exact:  # type: ignore[operator]
                break
            ok = True
            candidate_windows = compute_windows_for_S(S, L_base)
        else:
            candidate_windows = compute_windows_for_S(S, L_base)
            C_prime = C * int(sum(candidate_windows))
            if using_budget:
                ok = C_prime <= max_pseudochannels  # type: ignore[operator]
            else:
                expected_hits = N * _mu_k(C_prime) / float(C_prime)  # type: ignore[operator]
                ok = expected_hits >= H  # type: ignore[operator]

        if ok:
            best_S = S
            best_lengths = lengths[:S]
            best_windows = candidate_windows
            best_L_base = L_base
            constraint_met = True

    if not constraint_met:
        if using_budget:
            warnings.warn(
                (
                    f"Pseudochannel budget max_pseudochannels={max_pseudochannels} "
                    "could not be met even at S=1. Falling back to S=1."
                ),
                stacklevel=2,
            )
        elif using_coverage:
            warnings.warn(
                (
                    f"Coverage constraint H={H} could not be met even at S=1. "
                    "Falling back to S=1."
                ),
                stacklevel=2,
            )
    elif using_exact and best_S != S_exact:
        warnings.warn(
            (
                f"Requested S_exact={S_exact}, but only S={best_S} satisfies the "
                "mobility constraint. Using the largest feasible value."
            ),
            stacklevel=2,
        )

    return best_S, best_lengths, best_windows, best_L_base


@dataclass
class RomanOperator(BaseEstimator, TransformerMixin):
    """
    ROMAN (ROuting Multiscale representAtioN) routing operator.

    ROMAN is a deterministic front-end operator for time series. It builds an
    anti-aliased multiscale pyramid, extracts fixed-length overlapping windows
    from each scale, and stacks those windows as pseudochannels. The resulting
    representation makes temporal scale and coarse temporal position explicit in
    the channel structure while shortening the processed time axis to a common
    base length.

    This class implements the operator-level transformation studied in the
    paper. It is intended to be placed before a standard convolutional
    classifier rather than used as a classifier on its own.

    Inputs
    ------
    ROMAN accepts either:

    - univariate data with shape `(n_instances, n_timepoints)`
    - multivariate data with shape `(n_instances, n_variables, n_timepoints)`

    Outputs
    -------
    After fitting, `transform(X)` returns an array of shape:

    `(n_instances, C * sum_s W_s, L_base)`

    where:

    - `C` is the number of original channels
    - `W_s` is the number of windows extracted at scale `s`
    - `L_base` is the common window length set by the coarsest scale

    Notes
    -----
    - `S=1` is exactly the identity case.
    - When `normalization=True`, channel-wise mean and standard deviation are
      estimated from the training data before pyramid construction.
    - Downsampling uses the anti-aliasing filter `[1, 2, 1] / 4` followed by
      decimation by 2.
    - Odd-length scales are padded by repeating the last value so no sample is
      dropped before decimation.
    """

    alpha: float
    min_timesteps_per_channel: int
    normalization: bool = True
    window_rule: str = "overlap"
    window_surplus: int = 0
    S_max: Optional[int] = None

    S: Optional[int] = None
    max_pseudochannels: Optional[int] = None
    N: Optional[int] = None
    H: Optional[float] = None

    S_: Optional[int] = None
    L_: Optional[int] = None
    C_: Optional[int] = None
    L_base_: Optional[int] = None
    lengths_: Optional[List[int]] = None
    windows_: Optional[List[int]] = None
    starts_: Optional[List[np.ndarray]] = None
    ends_: Optional[List[np.ndarray]] = None
    n_pseudochannels_: Optional[int] = None

    mean_x_: Optional[np.ndarray] = None
    std_x_: Optional[np.ndarray] = None

    eps: float = 1e-8

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RomanOperator":
        """
        Fit ROMAN to the training data.

        The fitting step does not learn model weights. Instead, it:

        - validates the chosen scale-selection mode
        - infers the input dimensionality
        - selects the final number of scales and windows
        - precomputes window indices for fast later transforms
        - optionally stores normalization statistics
        """
        del y

        using_exact = self.S is not None
        using_budget = self.max_pseudochannels is not None
        using_coverage = (self.N is not None) or (self.H is not None)

        mode_count = sum([using_exact, using_budget, using_coverage])
        if mode_count != 1:
            raise ValueError(
                "Provide exactly one of S, max_pseudochannels, or the pair (N, H)."
            )

        if using_exact:
            if self.S is None or self.S <= 0:
                raise ValueError("S must be an integer >= 1.")
            N = None
            H = None
            max_pseudochannels = None
            S_exact = self.S
        elif using_budget:
            if self.max_pseudochannels is None or self.max_pseudochannels <= 0:
                raise ValueError("max_pseudochannels must be an integer >= 1.")
            N = None
            H = None
            max_pseudochannels = self.max_pseudochannels
            S_exact = None
        else:
            if self.N is None or self.H is None:
                raise ValueError("Coverage mode requires both N and H.")
            if self.N <= 0:
                raise ValueError("N must be >= 1.")
            if self.H <= 0:
                raise ValueError("H must be > 0.")
            N = self.N
            H = self.H
            max_pseudochannels = None
            S_exact = None

        X3 = self._ensure_3d(X)
        _, C, L = X3.shape

        S_out, lengths, windows, L_base = choose_S_roman(
            C=C,
            alpha=self.alpha,
            L=L,
            min_timesteps_per_channel=self.min_timesteps_per_channel,
            S_exact=S_exact,
            max_pseudochannels=max_pseudochannels,
            N=N,
            H=H,
            window_rule=self.window_rule,
            window_surplus=self.window_surplus,
            S_max=self.S_max,
        )

        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []
        for s in range(S_out):
            Ls = lengths[s]
            Ws = windows[s]
            start_idx, end_idx = self._compute_windows(L=Ls, W=Ws, win_len=L_base)
            starts.append(start_idx)
            ends.append(end_idx)

        if self.normalization:
            mean, std = self._fit_channel_norm(X3)
            self.mean_x_ = mean
            self.std_x_ = std
        else:
            self.mean_x_ = None
            self.std_x_ = None

        self.S_ = int(S_out)
        self.L_ = int(L)
        self.C_ = int(C)
        self.L_base_ = int(L_base)
        self.lengths_ = list(lengths)
        self.windows_ = list(windows)
        self.starts_ = starts
        self.ends_ = ends
        self.n_pseudochannels_ = int(C * sum(windows))

        return self

    def transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Apply the ROMAN transformation to a dataset.

        Parameters
        ----------
        X : np.ndarray
            Input dataset with the same channel count and sequence length used
            during `fit`.
        batch_size : int, default=1024
            Number of instances transformed at once.

        Returns
        -------
        np.ndarray
            ROMAN-transformed dataset with stacked pseudochannels. Each
            pseudochannel corresponds to an original channel, a pyramid scale,
            and a coarse temporal window.
        """
        self._check_is_fitted()

        X3 = self._ensure_3d(X).astype(np.float32, copy=False)
        n_instances, num_channels, seq_len = X3.shape

        if num_channels != self.C_:
            raise ValueError(
                f"transform expected {self.C_} channels, but received {num_channels}."
            )
        if seq_len != self.L_:
            raise ValueError(
                f"transform expected length {self.L_}, but received {seq_len}."
            )

        if self.normalization:
            X3 = self._apply_channel_norm(X3, self.mean_x_, self.std_x_)
        else:
            np.nan_to_num(X3, copy=False, nan=0.0)

        total_windows = int(sum(self.windows_))
        Z = np.empty(
            (n_instances, num_channels * total_windows, self.L_base_),
            dtype=np.float32,
        )

        for start in range(0, n_instances, batch_size):
            stop = start + batch_size
            X_batch = X3[start:stop]
            pyramid_batch = self._build_pyramid(X_batch, S=self.S_)

            out_idx = 0
            for s in range(self.S_):
                Xs = pyramid_batch[s]
                st = self.starts_[s]
                en = self.ends_[s]
                Ws = st.shape[0]

                for w in range(Ws):
                    a = int(st[w])
                    b = int(en[w])
                    Z[start:stop, out_idx:out_idx + num_channels, :] = Xs[:, :, a:b]
                    out_idx += num_channels

        return Z

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit ROMAN on `X` and immediately transform the same dataset."""
        return self.fit(X, y=y).transform(X)

    def map_relevance(self, relevance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map pseudochannel relevance back to the original timeline.

        This method is useful when the downstream model provides one relevance
        score per pseudochannel, for example linear coefficients from a
        classifier trained on ROMAN-transformed data.

        Parameters
        ----------
        relevance : np.ndarray
            One-dimensional array with one value per pseudochannel.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two arrays:

            - channel-time relevance with shape `(C, L)`
            - scale-time relevance with shape `(S, L)`
        """
        self._check_is_fitted()

        total_windows = int(sum(self.windows_))
        expected_len = self.C_ * total_windows
        if relevance.ndim != 1 or len(relevance) != expected_len:
            raise ValueError(
                f"Expected a relevance vector of length {expected_len}, got {relevance.shape}."
            )

        relevance_map_c = np.zeros((self.C_, self.L_), dtype=np.float32)
        counts_map_c = np.zeros((self.C_, self.L_), dtype=np.int32)
        relevance_map_s = np.zeros((self.S_, self.L_), dtype=np.float32)
        counts_map_s = np.zeros((self.S_, self.L_), dtype=np.int32)

        idx = 0
        for s in range(self.S_):
            stride = 2 ** s
            Ws = self.windows_[s]
            starts = self.starts_[s]
            ends = self.ends_[s]

            for w in range(Ws):
                orig_start = int(starts[w] * stride)
                orig_start = max(0, min(self.L_, orig_start))
                orig_end = int(ends[w] * stride)
                orig_end = max(0, min(self.L_, orig_end))

                for c in range(self.C_):
                    rel_val = relevance[idx]
                    if orig_end > orig_start:
                        rel_norm = rel_val / float(stride)
                        relevance_map_c[c, orig_start:orig_end] += rel_norm
                        counts_map_c[c, orig_start:orig_end] += 1
                        relevance_map_s[s, orig_start:orig_end] += rel_norm
                        counts_map_s[s, orig_start:orig_end] += 1
                    idx += 1

        rel_c = relevance_map_c / np.maximum(counts_map_c, 1)
        rel_s = relevance_map_s / np.maximum(counts_map_s, 1)
        return rel_c, rel_s

    def plot_relevance(
        self,
        relevance_map: np.ndarray,
        ylabel: str = "Channels",
        figsize: Tuple[int, int] = (10, 6),
        cmap: str = "viridis",
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot a relevance heatmap and simple marginals.

        Parameters
        ----------
        relevance_map : np.ndarray
            Two-dimensional relevance array, usually returned by
            `map_relevance`.
        ylabel : str, default="Channels"
            Label used for the vertical axis.
        figsize : tuple[int, int], default=(10, 6)
            Matplotlib figure size.
        cmap : str, default="viridis"
            Colormap used for the heatmap.
        save_path : str, optional
            Optional path used to save the figure.
        title : str, optional
            Figure title. When omitted, a default ROMAN title is used.
        """
        if title is None:
            title = f"ROMAN Positional & {ylabel} Relevance"

        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError as exc:
            raise ImportError(
                "matplotlib must be installed to use plot_relevance."
            ) from exc

        if relevance_map.ndim != 2:
            raise ValueError(
                f"Expected a 2D relevance map, got shape {relevance_map.shape}."
            )

        C, L = relevance_map.shape
        time_importance = relevance_map.sum(axis=0)
        fig = plt.figure(figsize=figsize)

        if C == 1:
            ax = fig.add_subplot(111)
            x_vals = np.arange(L)
            y_vals = time_importance
            norm = plt.Normalize(y_vals.min(), y_vals.max())
            color_map = plt.get_cmap(cmap)

            ax.plot(x_vals, y_vals, color="black", linewidth=1.2, alpha=0.8)
            for i in range(L - 1):
                mid_val = (y_vals[i] + y_vals[i + 1]) / 2.0
                ax.fill_between(
                    [x_vals[i], x_vals[i + 1]],
                    [y_vals[i], y_vals[i + 1]],
                    color=color_map(norm(mid_val)),
                    alpha=0.75,
                    linewidth=0,
                )

            ax.set_xlabel("Time Steps", fontsize=12, fontweight="bold")
            ax.set_ylabel("Relevance Score", fontsize=12, fontweight="bold")
            ax.set_xlim(0, L - 1)
            ax.set_ylim(0, y_vals.max() * 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if title:
                fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            return

        channel_importance = relevance_map.sum(axis=1)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[5, 1],
            height_ratios=[4, 1.2],
            wspace=0.05,
            hspace=0.08,
        )

        ax_main = fig.add_subplot(gs[0, 0])
        im = ax_main.imshow(
            relevance_map,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
        )
        ax_main.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax_main.tick_params(axis="x", labelbottom=False)
        if C <= 20:
            ax_main.set_yticks(np.arange(C))

        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
        x_vals = np.arange(L)
        ax_bottom.plot(x_vals, time_importance, color="black", linewidth=1.5)
        ax_bottom.fill_between(x_vals, time_importance, color="black", alpha=0.2)
        ax_bottom.set_xlabel("Time Steps", fontsize=12, fontweight="bold")
        ax_bottom.set_ylabel("Cumulative\nImportance", fontsize=10)
        ax_bottom.grid(True, linestyle="--", alpha=0.5)
        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)
        ax_bottom.set_xlim(-0.5, L - 0.5)

        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
        y_vals = np.arange(C)
        ax_right.barh(
            y_vals,
            channel_importance,
            color="black",
            alpha=0.7,
            height=1.0,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_right.set_xlabel("Cumulative\nImportance", fontsize=10)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(True, linestyle="--", alpha=0.5, axis="x")
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.set_ylim(-0.5, C - 0.5)

        cbar_ax = fig.add_axes([0.92, 0.40, 0.02, 0.45])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Relevance Score", rotation=270, labelpad=15, fontsize=10)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim == 2:
            warnings.warn(
                (
                    "Input is 2D, so ROMAN assumes shape "
                    "(n_instances, n_timepoints) and reshapes it to "
                    "(n_instances, 1, n_timepoints)."
                ),
                stacklevel=2,
            )
            return X[:, None, :]
        if X.ndim == 3:
            return X
        raise ValueError(
            "X must have shape (n_instances, n_timepoints) or "
            "(n_instances, n_variables, n_timepoints)."
        )

    def _fit_channel_norm(self, X3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.nanmean(X3, axis=(0, 2))
        std = np.nanstd(X3, axis=(0, 2), ddof=0)
        std = np.maximum(std, self.eps)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _apply_channel_norm(
        X3: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        X_norm = (X3 - mean[None, :, None]) / std[None, :, None]
        np.nan_to_num(X_norm, copy=False, nan=0.0)
        return X_norm

    def _check_is_fitted(self) -> None:
        if self.S_ is None or self.starts_ is None or self.ends_ is None:
            raise RuntimeError("RomanOperator is not fitted. Call fit(X) first.")

    @staticmethod
    def _pad_to_even_last(X: np.ndarray) -> np.ndarray:
        """Repeat the final time step when a scale has odd length."""
        L = X.shape[-1]
        if (L % 2) == 0:
            return X
        last = X[..., -1:]
        return np.concatenate([X, last], axis=-1)

    @staticmethod
    def _lowpass_and_decimate_by_2(X: np.ndarray) -> np.ndarray:
        """
        Apply `[1, 2, 1] / 4` low-pass filtering and decimate by a factor of 2.
        """
        X = RomanOperator._pad_to_even_last(X)
        _, _, L = X.shape

        left = X[:, :, 0:1]
        right = X[:, :, -1:]
        Xp = np.concatenate([left, X, right], axis=2)
        y = (Xp[:, :, 0:L] + 2.0 * Xp[:, :, 1 : L + 1] + Xp[:, :, 2 : L + 2]) * 0.25
        return y[:, :, ::2]

    def _build_pyramid(self, X3: np.ndarray, S: int) -> List[np.ndarray]:
        """Build the multiscale ROMAN pyramid up to scale `S`."""
        pyramid: List[np.ndarray] = [X3]
        for _ in range(2, S + 1):
            pyramid.append(self._lowpass_and_decimate_by_2(pyramid[-1]))
        return pyramid

    @staticmethod
    def _compute_windows(L: int, W: int, win_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute `W` fixed-length windows that span a sequence of length `L`.

        The first window always starts at 0 and the final window always ends at
        `L`. Any mismatch introduced by integer indexing is absorbed as extra
        overlap between neighboring windows.
        """
        if win_len <= 0:
            raise ValueError("win_len must be >= 1")
        if L <= 0:
            raise ValueError("L must be >= 1")
        if W <= 0:
            raise ValueError("W must be >= 1")

        if win_len > L:
            return np.array([0], dtype=np.int32), np.array([L], dtype=np.int32)

        if W == 1 or L == win_len:
            return np.array([0], dtype=np.int32), np.array([win_len], dtype=np.int32)

        target_advance = L - win_len
        starts = np.empty((W,), dtype=np.int32)
        starts[0] = 0
        for w in range(1, W):
            starts[w] = int(math.floor((w * target_advance) / (W - 1)))

        starts[-1] = L - win_len
        ends = starts + win_len
        starts = np.clip(starts, 0, L - win_len)
        ends = np.clip(ends, win_len, L)
        return starts, ends
