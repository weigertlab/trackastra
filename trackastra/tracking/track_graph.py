"""Adapted from https://github.com/funkelab/motile/blob/05fc67f1763afe806f244d10210fa66daa3dca67/motile/track_graph.py.

MIT License

Copyright (c) 2023 Funke lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


class TrackGraph(nx.DiGraph):
    """A :class:`networkx.DiGraph` of objects with positions in time and space,
    and inter-frame edges between them.

    Provides a few convenience methods for time series graphs in addition to
    all the methods inherited from :class:`networkx.DiGraph`.

    Args:
        graph_data (optional):

            Optional graph data to pass to the :class:`networkx.DiGraph`
            constructor as ``incoming_graph_data``. This can be used to
            populate a track graph with entries from a generic
            ``networkx`` graph.

        frame_attribute (``string``, optional):

            The name of the node attribute that corresponds to the frame (i.e.,
            the time dimension) of the object. Defaults to ``'t'``.
    """

    def __init__(self, graph_data=None, frame_attribute="t"):
        super().__init__(incoming_graph_data=graph_data)

        self.frame_attribute = frame_attribute
        self._graph_changed = True

        self._update_metadata()

    def prev_edges(self, node):
        """Get all edges that point forward into ``node``."""
        return self.in_edges(node)

    def next_edges(self, node):
        """Get all edges that point forward out of ``node``."""
        return self.out_edges(node)

    def get_frames(self):
        """Get a tuple ``(t_begin, t_end)`` of the first and last frame
        (exclusive) this track graph has nodes for.
        """
        self._update_metadata()

        return (self.t_begin, self.t_end)

    def nodes_by_frame(self, t):
        """Get all nodes in frame ``t``."""
        self._update_metadata()

        if t not in self._nodes_by_frame:
            return []
        return self._nodes_by_frame[t]

    def _update_metadata(self):
        if not self._graph_changed:
            return

        self._graph_changed = False

        if self.number_of_nodes() == 0:
            self._nodes_by_frame = {}
            self.t_begin = None
            self.t_end = None
            return

        self._nodes_by_frame = {}
        for node, data in self.nodes(data=True):
            t = data[self.frame_attribute]
            if t not in self._nodes_by_frame:
                self._nodes_by_frame[t] = []
            self._nodes_by_frame[t].append(node)

        frames = self._nodes_by_frame.keys()
        self.t_begin = min(frames)
        self.t_end = max(frames) + 1

        # ensure edges point forwards in time
        for u, v in self.edges:
            t_u = self.nodes[u][self.frame_attribute]
            t_v = self.nodes[v][self.frame_attribute]
            assert t_u < t_v, (
                f"Edge ({u}, {v}) does not point forwards in time, but from "
                f"frame {t_u} to {t_v}"
            )

        self._graph_changed = False

    # wrappers around node/edge add/remove methods:

    def add_node(self, n, **attr):
        super().add_node(n, **attr)
        self._graph_changed = True

    def add_nodes_from(self, nodes, **attr):
        super().add_nodes_from(nodes, **attr)
        self._graph_changed = True

    def remove_node(self, n):
        super().remove_node(n)
        self._graph_changed = True

    def remove_nodes_from(self, nodes):
        super().remove_nodes_from(nodes)
        self._graph_changed = True

    def add_edge(self, u, v, **attr):
        super().add_edge(u, v, **attr)
        self._graph_changed = True

    def add_edges_from(self, ebunch_to_add, **attr):
        super().add_edges_from(ebunch_to_add, **attr)
        self._graph_changed = True

    def add_weighted_edges_From(self, ebunch_to_add):
        super().add_weighted_edges_From(ebunch_to_add)
        self._graph_changed = True

    def remove_edge(self, u, v):
        super().remove_edge(u, v)
        self._graph_changed = True

    def update(self, edges, nodes):
        super().update(edges, nodes)
        self._graph_changed = True

    def clear(self):
        super().clear()
        self._graph_changed = True

    def clear_edges(self):
        super().clear_edges()
        self._graph_changed = True
