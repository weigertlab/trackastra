import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from trackastra.data import AugmentationPipeline


def plot_augs(b1, b2):
    import matplotlib.pyplot as plt

    x1, _y1, p1, t1 = b1
    x2, _y2, p2, t2 = b2
    plt.ion()
    fig = plt.figure(num=1)
    fig.clf()
    axs = fig.subplots(2, len(np.unique(t1)))
    for i, t in enumerate(np.unique(t1)):
        axs[0, i].imshow(x1[t], clim=(0, 1))
        axs[0, i].plot(*p1[t1 == t].T[::-1], "o", color="C2", alpha=0.4)
        axs[1, i].imshow(x2[t], clim=(0, 1))
        axs[1, i].plot(*p2[t2 == t].T[::-1], "o", color="C2", alpha=0.4)


def generate_data(ndim: int = 2):
    y = np.zeros((4,) + (64,) * ndim, np.uint16)

    points = np.stack(
        np.meshgrid(*tuple(np.arange(10, 50, 10) for _ in range(ndim)), indexing="ij"),
        axis=-1,
    ).reshape(-1, ndim)
    ts = np.random.randint(0, 4, len(points))
    for i in range(len(y)):
        y[i][tuple(points[ts == i].T)] = 1

    y = maximum_filter(y, size=(0,) + (3,) * ndim)
    x = y + 0.2 + 0.8 * np.random.uniform(0, 1, y.shape)
    return x, y, points, ts


@pytest.mark.skip(reason="outdated")
def test_augpipeline(plot=False):

    x, y, points, ts = generate_data()
    pipe = AugmentationPipeline(level=3, p=1)
    (x2, y2, p2), idx = pipe(x, y, points, ts)
    t2 = ts[idx]
    p2 = p2.astype(int)

    for i in range(len(y)):
        assert np.all(y2[i][tuple(p2[t2 == i].T)] == 1)

    if plot:
        plot_augs((x, y, points, ts), (x2, y2, p2, t2))


if __name__ == "__main__":

    test_augpipeline(plot=True)

    # pipe = RandomCrop((30, 40, 10, 20), ensure_inside_points=True)
    # pipe = RandomCrop((30,40), ensure_inside_points=True)
    # x, y, p, ts = generate_data()
    # (x2, y2, p2), idx = pipe(x, y, p)

    # pipe = AugmentationPipeline(level=4, p=1)

    # x, y, p, ts = generate_data(ndim=3)

    # (x2, y2, p2), idx = pipe(x, y, p, ts)

    # for _ in range(10):
    #     (x2, y2, p2), idx = pipe(x, y, p)
    #     print(len(idx))
