import colorsys
import itertools
import logging
import random
import sys
from pathlib import Path
from timeit import default_timer

import dask.array as da
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _single_color_integer_cmap(color=(0.3, 0.4, 0.5)):
    from matplotlib.colors import Colormap

    assert len(color) in (3, 4)

    class BinaryMap(Colormap):
        def __init__(self, color):
            self.color = np.array(color)
            if len(self.color) == 3:
                self.color = np.concatenate([self.color, [1]])

        def __call__(self, X, alpha=None, bytes=False):
            res = np.zeros((*X.shape, 4), np.float32)
            res[..., -1] = self.color[-1]
            res[X > 0] = np.expand_dims(self.color, 0)
            if bytes:
                return np.clip(256 * res, 0, 255).astype(np.uint8)
            else:
                return res

    return BinaryMap(color)


def render_label(
    lbl,
    img=None,
    cmap=None,
    cmap_img="gray",
    alpha=0.5,
    alpha_boundary=None,
    normalize_img=True,
):
    """Renders a label image and optionally overlays it with another image. Used for generating simple output images to asses the label quality.

    Parameters
    ----------
    lbl: np.ndarray of dtype np.uint16
        The 2D label image
    img: np.ndarray
        The array to overlay the label image with (optional)
    cmap: string, tuple, or callable
        The label colormap. If given as rgb(a)  only a single color is used, if None uses a random colormap
    cmap_img: string or callable
        The colormap of img (optional)
    alpha: float
        The alpha value of the overlay. Set alpha=1 to get fully opaque labels
    alpha_boundary: float
        The alpha value of the boundary (if None, use the same as for labels, i.e. no boundaries are visible)
    normalize_img: bool
        If True, normalizes the img (if given)

    Returns:
    -------
    img: np.ndarray
        the (m,n,4) RGBA image of the rendered label

    Example:
    -------
    from scipy.ndimage import label, zoom
    img = zoom(np.random.uniform(0,1,(16,16)),(8,8),order=3)
    lbl,_ = label(img>.8)
    u1 = render_label(lbl, img = img, alpha = .7)
    u2 = render_label(lbl, img = img, alpha = 0, alpha_boundary =.8)
    plt.subplot(1,2,1);plt.imshow(u1)
    plt.subplot(1,2,2);plt.imshow(u2)

    """
    from matplotlib import cm
    from skimage.segmentation import find_boundaries

    alpha = np.clip(alpha, 0, 1)

    if alpha_boundary is None:
        alpha_boundary = alpha

    if cmap is None:
        cmap = random_label_cmap()
    elif isinstance(cmap, tuple):
        cmap = _single_color_integer_cmap(cmap)
    else:
        pass

    cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_img = cm.get_cmap(cmap_img) if isinstance(cmap_img, str) else cmap_img

    # render image if given
    if img is None:
        im_img = np.zeros((*lbl.shape, 4), np.float32)
        im_img[..., -1] = 1

    else:
        assert lbl.shape[:2] == img.shape[:2]
        img = normalize(img) if normalize_img else img
        if img.ndim == 2:
            im_img = cmap_img(img)
        elif img.ndim == 3:
            im_img = img[..., :4]
            if img.shape[-1] < 4:
                im_img = np.concatenate(
                    [img, np.ones(img.shape[:2] + (4 - img.shape[-1],))], axis=-1
                )
        else:
            raise ValueError("img should be 2 or 3 dimensional")

    # render label
    im_lbl = cmap(lbl)

    mask_lbl = lbl > 0
    mask_bound = np.bitwise_and(mask_lbl, find_boundaries(lbl, mode="thick"))

    # blend
    im = im_img.copy()

    im[mask_lbl] = alpha * im_lbl[mask_lbl] + (1 - alpha) * im_img[mask_lbl]
    im[mask_bound] = (
        alpha_boundary * im_lbl[mask_bound] + (1 - alpha_boundary) * im_img[mask_bound]
    )

    return im


def random_label_cmap(n=2**16, h=(0, 1), lightness=(0.4, 1), s=(0.2, 0.8)):
    import matplotlib.colors

    h, lightness, s = (
        np.random.uniform(*h, n),
        np.random.uniform(*lightness, n),
        np.random.uniform(*s, n),
    )
    cols = np.stack(
        [colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, lightness, s)], axis=0
    )
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


# @torch.jit.script
def _blockwise_sum_with_bounds(A: torch.Tensor, bounds: torch.Tensor, dim: int = 0):
    A = A.transpose(dim, 0)
    cum = torch.cumsum(A, dim=0)
    cum = torch.cat((torch.zeros_like(cum[:1]), cum), dim=0)
    B = torch.zeros_like(A, device=A.device)
    for i, j in itertools.pairwise(bounds[:-1], bounds[1:]):
        B[i:j] = cum[j] - cum[i]
    B = B.transpose(0, dim)
    return B


def _bounds_from_timepoints(timepoints: torch.Tensor):
    assert timepoints.ndim == 1
    bounds = torch.cat((
        torch.tensor([0], device=timepoints.device),
        # torch.nonzero faster than torch.where
        torch.nonzero(timepoints[1:] - timepoints[:-1], as_tuple=False)[:, 0] + 1,
        torch.tensor([len(timepoints)], device=timepoints.device),
    ))
    return bounds


# def blockwise_sum(A: torch.Tensor, timepoints: torch.Tensor, dim: int = 0):
#     # get block boundaries
#     assert A.shape[dim] == len(timepoints)

#     bounds = _bounds_from_timepoints(timepoints)

#     # normalize within blocks
#     u = _blockwise_sum_with_bounds(A, bounds, dim=dim)
#     return u


def blockwise_sum(
    A: torch.Tensor, timepoints: torch.Tensor, dim: int = 0, reduce: str = "sum"
):
    if not A.shape[dim] == len(timepoints):
        raise ValueError(
            f"Dimension {dim} of A ({A.shape[dim]}) must match length of timepoints"
            f" ({len(timepoints)})"
        )

    A = A.transpose(dim, 0)

    if len(timepoints) == 0:
        logger.warning("Empty timepoints in block_sum. Returning zero tensor.")
        return A
    # -1 is the filling value for padded/invalid timepoints
    min_t = timepoints[timepoints >= 0]
    if len(min_t) == 0:
        logger.warning("All timepoints are -1 in block_sum. Returning zero tensor.")
        return A

    min_t = min_t.min()
    # after that, valid timepoints start with 1 (padding timepoints will be mapped to 0)
    ts = torch.clamp(timepoints - min_t + 1, min=0)
    index = ts.unsqueeze(1).expand(-1, len(ts))
    blocks = ts.max().long() + 1
    out = torch.zeros((blocks, A.shape[1]), device=A.device, dtype=A.dtype)
    out = torch.scatter_reduce(out, 0, index, A, reduce=reduce)
    B = out[ts]
    B = B.transpose(0, dim)

    return B


# TODO allow for batch dimension. Should be faster than looping
def blockwise_causal_norm(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    mode: str = "quiet_softmax",
    mask_invalid: torch.BoolTensor = None,
    eps: float = 1e-6,
):
    """Normalization over the causal dimension of A.

    For each block of constant timepoints, normalize the corresponding block of A
    such that the sum over the causal dimension is 1.

    Args:
        A (torch.Tensor): input tensor
        timepoints (torch.Tensor): timepoints for each element in the causal dimension
        mode: normalization mode.
            `linear`: Simple linear normalization.
            `softmax`: Apply exp to A before normalization.
            `quiet_softmax`: Apply exp to A before normalization, and add 1 to the denominator of each row/column.
        mask_invalid: Values that should not influence the normalization.
        eps (float, optional): epsilon for numerical stability.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    A = A.clone()

    if mode in ("softmax", "quiet_softmax"):
        # Subtract max for numerical stability
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        # TODO test without this subtraction

        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = -torch.inf
        # TODO set to min, then to 0 after exp

        # Blockwise max
        with torch.no_grad():
            ma0 = blockwise_sum(A, timepoints, dim=0, reduce="amax")
            ma1 = blockwise_sum(A, timepoints, dim=1, reduce="amax")

        u0 = torch.exp(A - ma0)
        u1 = torch.exp(A - ma1)

    elif mode == "linear":
        A = torch.sigmoid(A)
        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = 0

        u0, u1 = A, A
        ma0 = ma1 = 0
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    # get block boundaries and normalize within blocks
    # bounds = _bounds_from_timepoints(timepoints)
    # u0_sum = _blockwise_sum_with_bounds(u0, bounds, dim=0) + eps
    # u1_sum = _blockwise_sum_with_bounds(u1, bounds, dim=1) + eps

    u0_sum = blockwise_sum(u0, timepoints, dim=0) + eps
    u1_sum = blockwise_sum(u1, timepoints, dim=1) + eps

    if mode == "quiet_softmax":
        # Add 1 to the denominator of the softmax. With this, the softmax outputs can be all 0, if the logits are all negative.
        # If the logits are positive, the softmax outputs will sum to 1.
        # Trick: With maximum subtraction, this is equivalent to adding 1 to the denominator
        u0_sum += torch.exp(-ma0)
        u1_sum += torch.exp(-ma1)

    mask0 = timepoints.unsqueeze(0) > timepoints.unsqueeze(1)
    # mask1 = timepoints.unsqueeze(0) < timepoints.unsqueeze(1)
    # Entries with t1 == t2 are always masked out in final loss
    mask1 = ~mask0

    # blockwise diagonal will be normalized along dim=0
    res = mask0 * u0 / u0_sum + mask1 * u1 / u1_sum
    res = torch.clamp(res, 0, 1)

    return res


def blockwise_sum_batched(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    dim: int = -2,
    reduce: str = "sum",
):
    """Batched, loop-free equivalent of ``blockwise_sum`` over a batch dimension.

    Args:
        A: (B, N, N) tensor.
        timepoints: (B, N) tensor; -1 marks padded/invalid tokens.
        dim: -2/0 reduces over rows (within each row's timepoint block), -1/1 over
            columns. Matches the per-sample ``blockwise_sum`` convention.
        reduce: "sum" or "amax".

    Returns a (B, N, N) tensor; entry (b, i, j) holds the per-block reduction so the
    result is broadcast back to the original shape. Numerically identical to looping
    ``blockwise_sum`` over the batch (incl. the zeros-initialised amax, where the
    reduction is ``max(0, block_max)``), but without per-sample Python overhead or
    data-dependent CPU<->GPU syncs.
    """
    B, N, _ = A.shape
    rows = dim in (0, -2)

    if reduce == "sum":
        # same-timepoint membership mask M[b,i,j] = (t_i == t_j); padded tokens
        # (t=-1) group together exactly as the original block 0.
        M = (timepoints.unsqueeze(2) == timepoints.unsqueeze(1)).to(A.dtype)
        # rows: M @ A ; cols: A @ M (M is symmetric)
        return torch.bmm(M, A) if rows else torch.bmm(A, M)

    if reduce != "amax":
        raise ValueError(f"Unknown reduce {reduce}")

    # amax: replicate the original zeros-initialised scatter_reduce (=> max(0, .)).
    # Remap timepoints to dense block ids: valid -> 1.., padding -> 0 (own block),
    # matching ts = clamp(t - min_valid_t + 1, min=0). K = N + 1 bounds the ids
    # (<= #distinct timepoints + 1) without a data-dependent sync.
    t_valid = timepoints.masked_fill(timepoints < 0, torch.iinfo(torch.int64).max)
    min_t = t_valid.min(dim=1, keepdim=True).values
    ts = torch.clamp(timepoints - min_t + 1, min=0).long()
    K = N + 1

    if rows:
        index = ts.unsqueeze(-1).expand(B, N, N)  # block id along the row dim
        out = A.new_zeros((B, K, N))
        out.scatter_reduce_(1, index, A, reduce="amax", include_self=True)
        return out.gather(1, index)
    else:
        index = ts.unsqueeze(1).expand(B, N, N)  # block id along the col dim
        out = A.new_zeros((B, N, K))
        out.scatter_reduce_(2, index, A, reduce="amax", include_self=True)
        return out.gather(2, index)


def blockwise_causal_norm_batched(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    mode: str = "quiet_softmax",
    mask_invalid: torch.BoolTensor = None,
    eps: float = 1e-6,
):
    """Batched, loop-free equivalent of ``blockwise_causal_norm``.

    Args:
        A: (B, N, N) logits.
        timepoints: (B, N); -1 marks padded tokens.
        mode: "linear", "softmax" or "quiet_softmax".
        mask_invalid: (B, N, N) bool, entries excluded from the normalization.

    Numerically matches looping ``blockwise_causal_norm`` over the batch.
    """
    A = A.clone()

    if mode in ("softmax", "quiet_softmax"):
        if mask_invalid is not None:
            A[mask_invalid] = -torch.inf
        with torch.no_grad():
            ma0 = blockwise_sum_batched(A, timepoints, dim=0, reduce="amax")
            ma1 = blockwise_sum_batched(A, timepoints, dim=1, reduce="amax")
        u0 = torch.exp(A - ma0)
        u1 = torch.exp(A - ma1)
    elif mode == "linear":
        A = torch.sigmoid(A)
        if mask_invalid is not None:
            A[mask_invalid] = 0
        u0, u1 = A, A
        ma0 = ma1 = 0
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    u0_sum = blockwise_sum_batched(u0, timepoints, dim=0, reduce="sum") + eps
    u1_sum = blockwise_sum_batched(u1, timepoints, dim=1, reduce="sum") + eps

    if mode == "quiet_softmax":
        u0_sum = u0_sum + torch.exp(-ma0)
        u1_sum = u1_sum + torch.exp(-ma1)

    # mask0[b,i,j] = t_j > t_i (causal: child later than parent)
    mask0 = timepoints.unsqueeze(1) > timepoints.unsqueeze(2)
    mask1 = ~mask0

    res = mask0 * u0 / u0_sum + mask1 * u1 / u1_sum
    return torch.clamp(res, 0, 1)


def normalize_tensor(x: torch.Tensor, dim: int | None = None, eps: float = 1e-8):
    if dim is None:
        dim = tuple(range(x.ndim))

    mi, ma = torch.amin(x, dim=dim, keepdim=True), torch.amax(x, dim=dim, keepdim=True)
    return (x - mi) / (ma - mi + eps)


def normalize(x: np.ndarray | da.Array, subsample: int | None = 4):
    """Percentile normalize the image.

    If subsample is not None, calculate the percentile values over a subsampled image (last two axis)
    which is way faster for large images.
    """
    x = x.astype(np.float32)
    if subsample is not None and all(s > 64 * subsample for s in x.shape[-2:]):
        y = x[..., ::subsample, ::subsample]
    else:
        y = x

    mi, ma = np.percentile(y, (1, 99.8)).astype(np.float32)
    x -= mi
    x /= ma - mi + 1e-8
    return x


def seed(s=None):
    """Seed random number generators.

    Defaults to unix timestamp of function call.

    Args:
        s (``int``): Manual seed.
    """
    if s is None:
        s = int(default_timer())

    random.seed(s)
    logger.debug(f"Seed `random` rng with {s}.")
    np.random.seed(s)
    logger.debug(f"Seed `numpy` rng with {s}.")
    if "torch" in sys.modules:
        torch.manual_seed(s)
        logger.debug(f"Seed `torch` rng with {s}.")

    return s


def str2bool(x: str) -> bool:
    """Cast string to boolean.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    elif x.lower() in ("true", "t", "1"):
        return True
    elif x.lower() in ("false", "f", "0"):
        return False
    else:
        raise ValueError(f"'{x}' does not seem to be boolean.")


def str2path(x: str) -> Path:
    """Cast string to resolved absolute path.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    else:
        return Path(x).expanduser().resolve()


def none_or_str(x: str | None) -> str | None:
    """Allow for none value in string parsing."""
    if x in (None, "None", "none"):
        return None
    else:
        return str(x)


def none_or_path(x: str | None) -> Path | None:
    """Cast string to resolved absolute path or return None."""
    if x in (None, "None", "none"):
        return None
    else:
        return str2path(str(x))


if __name__ == "__main__":
    A = torch.rand(50, 50)
    idx = torch.tensor([0, 10, 20, A.shape[0]])

    A = torch.eye(50)

    B = _blockwise_sum_with_bounds(A, idx)

    tps = torch.repeat_interleave(torch.arange(5), 10)

    C = blockwise_causal_norm(A, tps)
