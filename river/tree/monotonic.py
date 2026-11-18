"""Monotonic constraint support for the online regression Hoeffding trees.

The coordinator keeps the piecewise-constant prediction function of a regression
Hoeffding tree monotonic along a set of constrained features, while every leaf keeps
learning freely from the incoming stream.

Design
------
Each tracked leaf owns a raw *region box* with a ``(lower, upper)`` interval per
constrained feature (unconstrained features do not appear). Two complementary kinds of
ordering constraints are maintained between leaves:

1. **Structure edges** come directly from the tree: every split on a constrained feature
   orders its two child subtrees in the constrained direction. They are created
   incrementally when a leaf splits.

2. **Region edges** are rebuilt from the current region boxes whenever the topology
   changes. A directed edge ``u -> v`` is emitted only when the boxes admit a strict,
   one-sided comparison (the lower region precedes the upper one on a constrained
   feature). Two-sided partial overlaps produced by independent split lines are not
   equalized, because doing so would merge regions that must keep their own fitted values
   and destroy the model's ability to learn. Leaves with identical constrained boxes that
   belong to the same component (e.g. children of an unconstrained-feature split that later
   drift apart) receive a bidirectional edge and are equalized by the isotonic fit.

3. **Tie pairs** handle the adaptive tree's main/alternate averaging explicitly: an
   alternate subtree root covering a branch region is tied to the branch's main leaves, so
   the averaged prediction stays consistent while the alternate has no split structure of
   its own.

The union graph is a DAG apart from the explicit tie and identical-box equalization
components. The coordinator fits an L2 isotonic regression on it by iteratively pooling
violated components, then derives per-leaf feasible intervals from the fitted neighbor
values. Leaf predictions are the raw running target means clamped to those intervals;
between coordination runs the means keep adapting, and a new run is triggered on a topology
change or when a touched mean escapes its cached interval.

Guarantee and scope
-------------------
For any two inputs whose leaf regions are strictly ordered (or equalized through a tie),
the predictions honor every constrained direction, including inputs the tree never observed
and points on either side of every split threshold. This covers the practical monotone
surface the tree can represent without collapsing independently learned regions; the
coordinator state stays proportional to the current main tree and the live alternate
subtrees, and no training history or pruned subtree is ever retained.
"""

from __future__ import annotations

import math

_INF = math.inf


def validate_constraint_config(monotonic_cst):
    """Validate a user provided ``monotonic_cst`` mapping.

    Returns an immutable copy (a tuple of ``(name, direction)`` pairs) or ``None`` when the
    configuration does not enable any constraint. The caller's mapping is never modified and
    the returned object does not share mutable state with it.
    """
    if monotonic_cst is None:
        return None

    if not isinstance(monotonic_cst, dict):
        raise ValueError(
            "'monotonic_cst' must be a dictionary mapping feature names to -1, 0 or 1."
        )

    normalized = {}
    for name, direction in monotonic_cst.items():
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Constrained feature names must be non-empty strings.")
        # bool is a subclass of int: reject it explicitly.
        if (
            isinstance(direction, bool)
            or not isinstance(direction, int)
            or direction not in (-1, 0, 1)
        ):
            raise ValueError(
                f"Invalid monotonic direction {direction!r} for feature {name!r}: "
                "only the Python integers -1, 0 and 1 are accepted."
            )
        if direction != 0:
            normalized[name] = direction

    if not normalized:
        return None

    return tuple(sorted(normalized.items()))


def validate_learn_input(x, y, w, constraints):
    """Validate a training sample when monotonic constraints are enabled."""
    _validate_x(x, constraints)

    if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y):
        raise ValueError("'y' must be a finite real number.")

    if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
        raise ValueError("'w' must be a finite, non-negative real number.")


def validate_predict_input(x, constraints):
    """Validate a prediction input when monotonic constraints are enabled."""
    _validate_x(x, constraints)


def _validate_x(x, constraints):
    if not isinstance(x, dict):
        raise ValueError("'x' must be a dictionary of feature values.")
    for name, _ in constraints:
        if name not in x:
            raise ValueError(f"Constrained feature {name!r} is missing from the input instance.")
        value = x[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Constrained feature {name!r} must hold a finite, non-boolean real number."
            )


class MonotonicTreeCoordinator:
    """Maintain monotonic feasible intervals for the leaves of a regression tree."""

    def __init__(self, constraints):
        self._constraints = constraints
        # Leaf -> raw region box, one (lower, upper) per constrained feature.
        self._boxes: dict = {}
        # Leaf -> feasible [lower, upper] prediction interval.
        self._bounds: dict = {}
        # Leaf -> component id. The main tree is component 0; every alternate subtree gets
        # its own component. Identical-box leaves inside one component are forced equal;
        # identical-box leaves across components are reconciled only through explicit ties.
        self._component: dict = {}
        self._next_component = 0
        # Monotonic creation sequence of each leaf. Object ids are reused across processes
        # and under garbage collection, so deterministic ordering uses this stable counter.
        self._order: dict = {}
        self._next_order = 0
        # Structure edges: leaf -> set of leaves constrained through a common constrained
        # split ancestor. Maintained incrementally across splits.
        self._struct_edges: dict = {}
        self._struct_reverse: dict = {}
        # Full constraint graph used by the isotonic fit (structure + region + ties).
        self._edges: dict = {}
        self._reverse: dict = {}
        # Bidirectional tie pairs (adaptive tree main/alternate consistency).
        self._ties: set = set()
        # The constraint graph only changes with the topology (splits, pruning, ties).
        # Mean-only updates reuse the cached graph and SCC decomposition and only re-fit
        # the isotonic program.
        self._graph_dirty = True
        # Cached SCC decomposition, invalidated together with the graph.
        self._component_of = {}
        self._components = []
        self._n_coordinations = 0

    @property
    def constraints(self):
        return self._constraints

    @property
    def n_coordinations(self):
        return self._n_coordinations

    @property
    def n_tracked_leaves(self):
        return len(self._boxes)

    # ------------------------------------------------------------------
    # Leaf lifecycle
    # ------------------------------------------------------------------
    def register_leaf(self, leaf, box=None, component=0):
        """Start tracking a brand new leaf inside the given constraint component."""
        if box is None:
            box = {name: (-_INF, _INF) for name, _ in self._constraints}
        self._boxes[leaf] = box
        self._bounds[leaf] = (-_INF, _INF)
        self._component[leaf] = component
        self._order[leaf] = self._next_order
        self._next_order += 1
        self._struct_edges[leaf] = set()
        self._struct_reverse[leaf] = set()
        self._edges[leaf] = set()
        self._reverse[leaf] = set()
        self._graph_dirty = True

    def new_component(self):
        """Reserve and return a component id for a new alternate subtree."""
        self._next_component += 1
        return self._next_component

    def split_leaf(self, parent, left_leaf, right_leaf, feature, threshold):
        """Register the replacement of ``parent`` by two children."""
        parent_box = self._boxes.pop(parent)
        parent_component = self._component.pop(parent)
        self._order.pop(parent, None)
        parent_ties = {pair for pair in self._ties if pair[0] is parent or pair[1] is parent}
        self._struct_edges.pop(parent, None)
        self._struct_reverse.pop(parent, None)
        self._bounds.pop(parent, None)
        self._edges.pop(parent, None)
        self._reverse.pop(parent, None)
        self._ties = {pair for pair in self._ties if pair not in parent_ties}

        left_box = {name: bounds for name, bounds in parent_box.items()}
        right_box = {name: bounds for name, bounds in parent_box.items()}

        direction = self._direction_of(feature)
        if direction is not None:
            low, high = parent_box[feature]
            left_box[feature] = (low, threshold)
            right_box[feature] = (threshold, high)

        self.register_leaf(left_leaf, left_box, component=parent_component)
        self.register_leaf(right_leaf, right_box, component=parent_component)

        if direction is not None:
            # Only the direct split ordering is stored; transitive ordering is rebuilt
            # from the region boxes whenever the topology changes.
            if direction == 1:
                self._add_struct_edge(left_leaf, right_leaf)
            else:
                self._add_struct_edge(right_leaf, left_leaf)

        # Inherit adaptive averaging ties.
        for pair in parent_ties:
            other = pair[0] if pair[1] is parent else pair[1]
            self._ties.add(self._tie_pair(left_leaf, other))
            self._ties.add(self._tie_pair(right_leaf, other))

        self._graph_dirty = True
        self.coordinate()

    def forget_subtree(self, leaves):
        """Remove leaves that no longer participate in any prediction (pruning)."""
        if not leaves:
            return
        leaves = set(leaves)
        for leaf in leaves:
            if leaf not in self._boxes:
                continue
            self._boxes.pop(leaf, None)
            self._bounds.pop(leaf, None)
            self._component.pop(leaf, None)
            self._order.pop(leaf, None)
            for successor in self._struct_edges.pop(leaf, set()):
                self._struct_reverse.get(successor, set()).discard(leaf)
            for predecessor in self._struct_reverse.pop(leaf, set()):
                self._struct_edges.get(predecessor, set()).discard(leaf)
        self._ties = {
            pair for pair in self._ties if pair[0] not in leaves and pair[1] not in leaves
        }
        self._graph_dirty = True
        self.coordinate()

    def add_tie(self, u, v):
        """Force two tracked leaves to share a single prediction value."""
        if u is v or u not in self._boxes or v not in self._boxes:
            return
        pair = self._tie_pair(u, v)
        if pair not in self._ties:
            self._ties.add(pair)
            self._graph_dirty = True
            self.coordinate()

    # ------------------------------------------------------------------
    # Feasible values
    # ------------------------------------------------------------------
    def clamp(self, leaf, value):
        """Project a raw leaf prediction onto the leaf's monotonic feasible interval."""
        if leaf not in self._bounds:
            # Self-healing for leaves reaching prediction before explicit registration.
            self.register_leaf(leaf)
            self.coordinate()
        lower, upper = self._bounds[leaf]
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    def needs_coordination(self, leaf, raw_mean):
        lower, upper = self._bounds[leaf]
        return raw_mean < lower or raw_mean > upper

    def coordinate(self):
        """Fit the isotonic program and recompute every leaf's feasible interval.

        The constraint graph and its strongly connected components only change with the
        topology; mean-only coordinations reuse the cached decomposition and pay just the
        isotonic pooling and interval propagation.
        """
        if not self._boxes:
            self._n_coordinations += 1
            return
        if self._graph_dirty:
            self._rebuild_graph()
            self._component_of, self._components, _ = self._tarjan_scc(self._ordered_leaves())
        component_of = self._component_of
        components = self._components
        leaves = self._ordered_leaves()

        # Maintain each pool's sum and count so its mean is an O(1) lookup instead of a
        # recomputation over all members on every pooling iteration.
        def sum_count(members):
            return sum(leaf.stats.mean.get() for leaf in members), len(members)

        pool_sum = {}
        pool_count = {}
        for cid, members in enumerate(components):
            s, n = sum_count(members)
            pool_sum[cid] = s
            pool_count[cid] = n

        def pool_value(cid):
            return pool_sum[cid] / pool_count[cid]

        # --- 2. Isotonic fit on the condensation DAG ------------------------------------
        condensation = {cid: set() for cid in pool_sum}
        for u in leaves:
            cu = component_of[u]
            for v in self._edges[u]:
                cv = component_of[v]
                if cv != cu:
                    condensation[cu].add(cv)

        pool_of = {cid: cid for cid in pool_sum}
        pool_members = {cid: list(components[cid]) for cid in pool_sum}
        pool_edges = {cid: set(edges) for cid, edges in condensation.items()}

        while True:
            violated = None
            for u in sorted(
                pool_edges,
                key=lambda cid: (
                    min(self._order[x] for x in pool_members[cid])
                    if pool_members[cid]
                    else float("inf")
                ),
            ):
                successors = pool_edges[u]
                if not pool_members[u]:
                    continue
                value_u = pool_value(u)
                for v in sorted(successors):
                    if pool_members[v] and value_u > pool_value(v) + 1e-15:
                        violated = (u, v)
                        break
                if violated is not None:
                    break
            if violated is None:
                break

            u, v = violated
            new_id = len(pool_members)
            pool_members[new_id] = pool_members[u] + pool_members[v]
            pool_sum[new_id] = pool_sum[u] + pool_sum[v]
            pool_count[new_id] = pool_count[u] + pool_count[v]
            pool_members[u] = []
            pool_members[v] = []
            for leaf in pool_members[new_id]:
                pool_of[component_of[leaf]] = new_id

            successors = (pool_edges.get(u, set()) | pool_edges.get(v, set())) - {u, v}
            predecessors = {
                node for node, node_succ in pool_edges.items() if u in node_succ or v in node_succ
            }
            for node in predecessors | {u, v}:
                pool_edges[node] = {s for s in pool_edges.get(node, set()) if s not in (u, v)}
            pool_edges[new_id] = successors
            for node in predecessors:
                if node not in (u, v):
                    pool_edges.setdefault(node, set()).add(new_id)

        fitted = {cid: pool_value(cid) for cid in pool_members if pool_members[cid]}

        # --- 3. Feasible intervals --------------------------------------------------------
        lower_bounds, upper_bounds = self._propagate_bounds(
            leaves, component_of, pool_of, fitted, components, pool_members
        )
        for leaf in leaves:
            self._bounds[leaf] = (lower_bounds[leaf], upper_bounds[leaf])

        self._n_coordinations += 1

    def _tarjan_scc(self, leaves):
        """Iterative Tarjan SCC over the current constraint graph."""
        index_counter = [0]
        stack = []
        on_stack = set()
        indices = {}
        lowlink = {}
        components = []
        component_of = {}

        for start in leaves:
            if start in indices:
                continue
            work = [
                (
                    start,
                    iter(sorted(self._edges.get(start, ()), key=lambda leaf: self._order[leaf])),
                )
            ]
            indices[start] = lowlink[start] = index_counter[0]
            index_counter[0] += 1
            stack.append(start)
            on_stack.add(start)

            while work:
                node, it = work[-1]
                advanced = False
                for nxt in it:
                    if nxt not in indices:
                        indices[nxt] = lowlink[nxt] = index_counter[0]
                        index_counter[0] += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append(
                            (
                                nxt,
                                iter(
                                    sorted(
                                        self._edges.get(nxt, ()), key=lambda leaf: self._order[leaf]
                                    )
                                ),
                            )
                        )
                        advanced = True
                        break
                    if nxt in on_stack:
                        lowlink[node] = min(lowlink[node], indices[nxt])
                if advanced:
                    continue

                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == indices[node]:
                    cid = len(components)
                    members = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component_of[member] = cid
                        members.append(member)
                        if member is node:
                            break
                    components.append(members)

        condensation = {cid: set() for cid in range(len(components))}
        for u, successors in self._edges.items():
            cu = component_of[u]
            for v in successors:
                cv = component_of[v]
                if cv != cu:
                    condensation[cu].add(cv)

        return component_of, components, condensation

    def _propagate_bounds(self, leaves, component_of, pool_of, fitted, components, pool_members):
        """Derive feasible leaf intervals from the fitted isotonic pool values.

        For a singleton pool the interval spans the largest fitted value reachable through
        any predecessor path and the smallest fitted value reachable through any successor
        path: clamping every leaf to such an interval guarantees that *every* constraint
        edge holds for *every* combination of in-interval predictions, even while the raw
        means drift between two coordination runs. Pools contracted from strongly connected
        components (tie pairs) are pinned to their common fitted value.
        """
        pool_ids = sorted({pool_of[component_of[leaf]] for leaf in leaves})
        adjacency: dict = {pid: set() for pid in pool_ids}
        reverse: dict = {pid: set() for pid in pool_ids}
        for u in leaves:
            pu = pool_of[component_of[u]]
            for v in self._edges[u]:
                pv = pool_of[component_of[v]]
                if pv == pu or pv not in adjacency:
                    continue
                if pv not in adjacency[pu]:
                    adjacency[pu].add(pv)
                    reverse[pv].add(pu)

        # Kahn topological order of the (acyclic) condensation. Pool ids are processed in
        # ascending order so the propagation is independent of set/hash iteration order.
        indegree = {pid: len(reverse[pid]) for pid in pool_ids}
        ready = sorted(pid for pid in pool_ids if indegree[pid] == 0)
        order = []
        while ready:
            pid = ready.pop(0)
            order.append(pid)
            for nxt in sorted(adjacency[pid]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
            ready.sort()

        # Longest-path lower bounds and shortest-path upper bounds over fitted values.
        lower_pool = {}
        upper_pool = {}
        pinned = {
            pid
            for pid in pool_ids
            if any(
                len(components[cid]) > 1
                for cid in {component_of[leaf] for leaf in pool_members[pid]}
            )
        }
        for pid in order:
            own = fitted[pid]
            if pid in pinned:
                lower_pool[pid] = own
            else:
                pred_max = max((lower_pool[p] for p in reverse[pid]), default=-_INF)
                lower_pool[pid] = pred_max
        for pid in reversed(order):
            own = fitted[pid]
            if pid in pinned:
                upper_pool[pid] = own
            else:
                succ_min = min((upper_pool[s] for s in adjacency[pid]), default=_INF)
                upper_pool[pid] = succ_min

        lower_bounds = {}
        upper_bounds = {}
        for leaf in leaves:
            pid = pool_of[component_of[leaf]]
            low = lower_pool[pid]
            high = upper_pool[pid]
            if low > high:
                low = high = 0.5 * (low + high)
            lower_bounds[leaf] = low
            upper_bounds[leaf] = high
        return lower_bounds, upper_bounds

    def _ordered_leaves(self):
        """Live leaves in deterministic creation order (stable across processes)."""
        return sorted(self._boxes, key=lambda leaf: self._order[leaf])

    # ------------------------------------------------------------------
    # Graph maintenance
    # ------------------------------------------------------------------
    def _add_struct_edge(self, u, v):
        if v not in self._struct_edges.setdefault(u, set()):
            self._struct_edges[u].add(v)
            self._struct_reverse.setdefault(v, set()).add(u)

    def _direction_of(self, feature):
        for name, direction in self._constraints:
            if name == feature:
                return direction
        return None

    def _rebuild_graph(self):
        """Combine structure edges, region ordering edges and tie pairs."""
        self._edges = {leaf: set() for leaf in self._boxes}
        self._reverse = {leaf: set() for leaf in self._boxes}
        self._graph_dirty = False
        self._component_of = {}
        self._components = []

        # 1. Incrementally maintained structure edges (pruned endpoints filtered out).
        for u, successors in self._struct_edges.items():
            if u not in self._boxes:
                continue
            for v in successors:
                if v in self._boxes:
                    self._edges[u].add(v)
                    self._reverse[v].add(u)

        # 2. Region ordering edges. A leaf precedes another whenever some point of its
        #    region precedes some point of the other's region. Leaves with identical
        #    constrained boxes are reachable by the same constrained assignments: when
        #    they belong to the same prediction surface (main tree or one alternate
        #    subtree) monotonicity forces a common value and they get a bidirectional edge;
        #    main/alternate pairs with identical boxes are reconciled through explicit
        #    ties instead.
        leaves = self._ordered_leaves()
        for i, u in enumerate(leaves):
            for v in leaves[i + 1 :]:
                self._add_region_edge(u, v)
                self._add_region_edge(v, u)

        # 3. Explicit ties as bidirectional edges.
        for u, v in self._ties:
            if u in self._boxes and v in self._boxes:
                self._edges[u].add(v)
                self._edges[v].add(u)
                self._reverse[u].add(v)
                self._reverse[v].add(u)

    def _add_region_edge(self, u, v):
        """Add the region ordering edge between two leaves when geometry allows it.

        The edge direction is derived solely from the region boxes, never from the call
        order, which makes the result independent of duplicate leaves sharing an interval:

        * identical boxes (regions differing only through unconstrained features) inside the
          same component are equalized through a bidirectional edge; cross-component
          identical boxes are reconciled through explicit adaptive ties;
        * strict one-sided comparisons produce a directed edge from the lower region to
          the upper one;
        * two-sided overlaps without a strict side produce no edge: threshold-adjacent
          regions are already linked by the split structure edges, and wider overlaps from
          independent split lines cannot be ordered without collapsing the function.
        """
        if self._identical_boxes(self._boxes[u], self._boxes[v]):
            if self._component[u] == self._component[v]:
                self._edges[u].add(v)
                self._edges[v].add(u)
                self._reverse[u].add(v)
                self._reverse[v].add(u)
            return

        directed = self._geometric_direction(u, v)
        if directed is None:
            return
        low_leaf, high_leaf = directed
        if high_leaf not in self._edges[low_leaf]:
            self._edges[low_leaf].add(high_leaf)
            self._reverse[high_leaf].add(low_leaf)

    def _geometric_direction(self, u, v):
        """Return ``(lower_leaf, upper_leaf)`` or ``None`` for unordered overlap.

        Only strict one-sided comparisons produce an edge: every point of the lower region
        must precede every point of the upper region on at least one constrained feature
        while the other features are merely compatible. Two-sided overlaps are skipped:
        threshold-adjacent ones are already linked through the split's structure edges, and
        wider overlaps come from independent split lines where an edge would either cycle
        or collapse regions that must keep their own fitted values.
        """
        u_first = None
        for name, direction in self._constraints:
            low_u, high_u = self._boxes[u][name]
            low_v, high_v = self._boxes[v][name]
            if direction == 1:
                u_below_v = high_u <= low_v
                v_below_u = high_v <= low_u
            else:
                u_below_v = high_v <= low_u
                v_below_u = high_u <= low_v
            if u_below_v and v_below_u:
                return None
            if u_below_v:
                if u_first is False:
                    return None
                u_first = True
            elif v_below_u:
                if u_first is True:
                    return None
                u_first = False
        if u_first is None:
            # The regions only overlap, without any strict one-sided feature comparison.
            return None
        return (u, v) if u_first else (v, u)

    @staticmethod
    def _same_bound(x, y):
        if x == y:
            return True
        scale = max(1.0, abs(x), abs(y))
        return abs(x - y) <= 1e-12 * scale

    def _identical_boxes(self, box_a, box_b):
        for name, _ in self._constraints:
            if box_a[name] != box_b[name]:
                return False
        return True

    def _boxes_fully_ordered(self, box_a, box_b):
        """True when every point of ``box_a`` precedes every point of ``box_b``."""
        for name, direction in self._constraints:
            low_a, high_a = box_a[name]
            low_b, high_b = box_b[name]
            if direction == 1:
                if high_a > low_b:
                    return False
            else:
                if high_b > low_a:
                    return False
        return True

    def _regions_have_ordered_points(self, box_a, box_b):
        """True when some point of ``box_a`` precedes some point of ``box_b``.

        For a ``+1`` feature an ordered pair ``x_a <= x_b`` exists while the smallest value
        of ``box_b`` does not strictly exceed the largest of ``box_a`` (``lower_b <=
        upper_a``); for a ``-1`` feature the condition is reversed. Threshold-adjacent
        regions that only touch at a split boundary stay comparable through equality.
        """
        for name, direction in self._constraints:
            low_a, high_a = box_a[name]
            low_b, high_b = box_b[name]
            if direction == 1:
                if low_b > high_a:
                    return False
            else:
                if low_a > high_b:
                    return False
        return True

    def hull_box(self, leaves):
        """Smallest box containing all given leaves' boxes."""
        leaves = [leaf for leaf in leaves if leaf in self._boxes]
        if not leaves:
            return None
        hull = {}
        for name, _ in self._constraints:
            low = min(self._boxes[leaf][name][0] for leaf in leaves)
            high = max(self._boxes[leaf][name][1] for leaf in leaves)
            hull[name] = (low, high)
        return hull

    def _tie_pair(self, u, v):
        return tuple(sorted((u, v), key=lambda leaf: self._order[leaf]))

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self):
        return {
            "_constraints": self._constraints,
            "_boxes": self._boxes,
            "_bounds": self._bounds,
            "_component": self._component,
            "_next_component": self._next_component,
            "_order": self._order,
            "_next_order": self._next_order,
            "_struct_edges": self._struct_edges,
            "_struct_reverse": self._struct_reverse,
            "_ties": self._ties,
            "_n_coordinations": self._n_coordinations,
        }

    def __setstate__(self, state):
        self._constraints = state["_constraints"]
        self._boxes = state["_boxes"]
        self._bounds = state["_bounds"]
        self._component = state.get("_component", {leaf: 0 for leaf in self._boxes})
        self._next_component = state.get("_next_component", 0)
        # Backward-compatible ordering: older pickles get id-based order on load.
        self._order = state.get("_order", {leaf: idx for idx, leaf in enumerate(self._boxes)})
        self._next_order = state.get("_next_order", len(self._order))
        self._struct_edges = state["_struct_edges"]
        self._struct_reverse = state["_struct_reverse"]
        self._ties = state.get("_ties", set())
        self._n_coordinations = state.get("_n_coordinations", 0)
        # Edges are derived state: rebuild lazily on the first coordination.
        self._edges = {}
        self._reverse = {}
        self._graph_dirty = True

    def rebuild(self):
        """Rebuild the constraint graph and re-coordinate (self-healing after unpickle)."""
        self._graph_dirty = True
        self.coordinate()
