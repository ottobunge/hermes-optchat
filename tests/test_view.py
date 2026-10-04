from optchat.view import PLACEHOLDER, View


class FakeTree:
    """node(l, i) -> text or None."""

    def __init__(self, nodes=None):
        self.nodes = dict(nodes or {})

    def node(self, l, i):
        return self.nodes.get((l, i))


def test_render_lines_use_message_ids_and_flatten_newlines():
    tree = FakeTree({(0, 0): "user: a\nb"})
    v = View(budget=10_000)
    v.append(0, tree)
    v.append(1, tree)
    assert v.render(tree) == "<chat>\n0+1|user: a b\n1+1|" + PLACEHOLDER + "\n</chat>"


def _hundred(c):
    return c * 100


def test_fit_merges_the_most_due_aligned_pair_with_built_parent():
    tree = FakeTree({(0, k): _hundred(str(k)) for k in range(4)})
    tree.nodes[(1, 0)] = "x" * 50
    tree.nodes[(1, 1)] = "y" * 50
    v = View(budget=350)
    for k in range(4):
        v.append(k, tree)
    v.fit(4, tree)
    assert v.parts == [(1, 0), (0, 2), (0, 3)]
    assert v.size(tree) == 250


def test_fit_waits_when_no_parent_is_built():
    tree = FakeTree({(0, k): _hundred("a") for k in range(4)})
    v = View(budget=100)
    for k in range(4):
        v.append(k, tree)
    v.fit(4, tree)
    assert v.parts == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_fit_never_merges_unaligned_or_mixed_level_neighbours():
    tree = FakeTree({(0, k): _hundred("a") for k in range(3)})
    tree.nodes[(1, 1)] = "b"  # parent of (0,2),(0,3): (0,3) is not in the view
    v = View(budget=10)
    v.parts = [(0, 0), (0, 1), (0, 2)]
    v.fit(3, tree)  # (0,1)+(0,2) are adjacent but not siblings
    assert v.parts == [(0, 0), (0, 1), (0, 2)]
    tree.nodes[(1, 0)] = "c"
    tree.nodes[(0, 3)] = "d"
    v.parts = [(1, 0), (0, 2)]
    v.fit(3, tree)  # mixed levels never merge
    assert v.parts == [(1, 0), (0, 2)]


def test_size_counts_utf8_bytes_and_placeholders():
    tree = FakeTree({(0, 0): "ñ" * 10})
    v = View(budget=1000)
    v.append(0, tree)
    v.append(1, tree)
    assert v.size(tree) == 20 + len(PLACEHOLDER)


def test_due_rule_prefers_older_relative_to_size():
    # Level-1 pair at 0..3 (weight 8) vs level-0 pair at 4,5 (weight 4), T = 6.
    # due(level1 @0) = 6/8 = 0.75 ; due(level0 @4) = 2/4 = 0.5  -> merge the level-1 pair.
    tree = FakeTree({(1, 0): "a" * 100, (1, 1): "b" * 100, (2, 0): "c",
                     (0, 4): "d" * 100, (0, 5): "e" * 100, (1, 2): "f"})
    v = View(budget=350)
    v.parts = [(1, 0), (1, 1), (0, 4), (0, 5)]
    v.fit(6, tree)
    assert v.parts == [(2, 0), (0, 4), (0, 5)]


def test_first_unbuilt_and_settled():
    tree = FakeTree({(0, 0): "a", (0, 1): "b"})
    v = View(budget=1000)
    for k in range(3):
        v.append(k, tree)
    assert v.first_unbuilt(3, tree) == 2
    assert not v.settled(tree)
    tree.nodes[(0, 2)] = "c"
    assert v.first_unbuilt(3, tree) == 3
    assert v.settled(tree)


def test_replay_fold_matches_incremental_history():
    import random

    rnd = random.Random(7)
    T = 300
    tree = FakeTree()
    for k in range(T):
        tree.nodes[(0, k)] = "m" * rnd.randint(20, 300)
    for l in range(1, 9):
        for i in range(T >> l):
            tree.nodes[(l, i)] = "p" * rnd.randint(100, 500)
    live = View(budget=8000)
    for k in range(T):
        live.append(k, tree)
        live.fit(k + 1, tree)
    replay = View.fold(T, tree, budget=8000)
    assert replay.parts == live.parts
    assert replay.size(tree) <= 8000
    # tiles [0, T) in order with no gaps
    pos = 0
    for l, i in replay.parts:
        assert i << l == pos
        pos += 1 << l
    assert pos == T


def test_replay_follows_the_recorded_completion_order():
    tree = FakeTree({(0, k): _hundred(str(k)) for k in range(4)})
    tree.nodes[(1, 0)] = "A" * 100
    tree.nodes[(1, 1)] = "B" * 100
    events = [("message", k) for k in range(4)] + [("node", 0, k) for k in range(4)]
    events += [("node", 1, 1), ("node", 1, 0)]  # (1,1) completed first
    v = View.replay(events, tree, budget=350)
    assert v.parts == [(0, 0), (0, 1), (1, 1)]
    assert v.size(tree) == 300
    # all nodes known up front (the old fold) resolves differently
    assert View.fold(4, tree, budget=350).parts == [(1, 0), (0, 2), (0, 3)]
