"""The view: tree nodes tiling the whole chat, oldest first, under a byte budget (spec §5)."""

from __future__ import annotations

from .store import nbytes

PLACEHOLDER = "(not summarized yet: zoom it)"


def flat(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


class _Known:
    """The tree as of some point in a replay: only nodes already committed."""

    def __init__(self, tree):
        self.tree = tree
        self.seen: set[tuple[int, int]] = set()

    def node(self, l: int, i: int):
        return self.tree.node(l, i) if (l, i) in self.seen else None


class View:
    def __init__(self, budget: int):
        self.budget = budget
        self._parts: list[tuple[int, int]] = []
        self._size = 0  # bytes of all part texts, kept incrementally

    @property
    def parts(self) -> list[tuple[int, int]]:
        return self._parts

    @parts.setter
    def parts(self, value) -> None:
        self._parts = list(value)
        self._size = None  # recomputed on next size()

    @classmethod
    def fold(cls, T: int, tree, budget: int) -> "View":
        """Rebuild the view at load: the same append + fit for every message in order."""
        v = cls(budget)
        for i in range(T):
            v.append(i, tree)
            v.fit(i + 1, tree)
        return v

    @classmethod
    def replay(cls, events, tree, budget: int) -> "View":
        """Rebuild the live view at load by redoing its history in commit order.

        ``events`` is ``Store.events()``: a message is appended then fit, a node becomes
        visible then fit, exactly as ``Memory.log`` / ``Memory._commit`` did live. Nodes
        built later are hidden until their event, so they cannot change past merges.
        """
        v, known, T = cls(budget), _Known(tree), 0
        for e in events:
            if e[0] == "message":
                T = e[1] + 1
                v.append(e[1], known)
            else:
                part = (e[1], e[2])
                old = v.text(part, known)
                known.seen.add(part)
                v.node_built(*part, old, known)
            v.fit(T, known)
        return v

    def first_unbuilt(self, T: int, tree) -> int:
        """First message whose view line is not a summary yet (``T`` if none)."""
        for l, i in self.parts:
            if tree.node(l, i) is None:
                return i << l
        return T

    def settled(self, tree) -> bool:
        return all(tree.node(l, i) is not None for l, i in self.parts)

    def append(self, i: int, tree) -> None:
        self._parts.append((0, i))
        if self._size is not None:
            self._size += nbytes(self.text((0, i), tree))

    def size(self, tree) -> int:
        if self._size is None:
            self._size = sum(nbytes(self.text(p, tree)) for p in self._parts)
        return self._size

    def node_built(self, l: int, i: int, old_text: str, tree) -> None:
        """A node's text appeared; adjust the size if it is a part (its placeholder was counted)."""
        if self._size is not None and l == 0 and self._parts and self._parts[-1][1] >= i:
            # only level-0 parts can be unbuilt; they sit at the end, after any merged parts
            for part in reversed(self._parts):
                if part == (l, i):
                    self._size += nbytes(self.text(part, tree)) - nbytes(old_text)
                    return
                if part[0] != 0 or part[1] < i:
                    return

    def fit(self, T: int, tree) -> None:
        """Merge the most due aligned sibling pair whose parent is built, until under budget.

        Never splits; parents not built yet are passed over (the view may stay over budget).
        """
        size = self.size(tree)
        while size > self.budget:
            best, best_due = None, -1.0
            for k in range(len(self.parts) - 1):
                (la, ia), (lb, ib) = self.parts[k], self.parts[k + 1]
                if la != lb or ia % 2 or ib != ia + 1 or tree.node(la + 1, ia // 2) is None:
                    continue
                due = (T - (ia << la)) / (1 << (la + 2))
                if due > best_due:
                    best, best_due = k, due
            if best is None:
                break
            (l, i), b = self.parts[best], self.parts[best + 1]
            parent = (l + 1, i // 2)
            size += nbytes(self.text(parent, tree)) - nbytes(self.text((l, i), tree)) - nbytes(self.text(b, tree))
            self._parts[best:best + 2] = [parent]
            self._size = size

    def text(self, part, tree) -> str:
        t = tree.node(*part)
        return PLACEHOLDER if t is None else t

    def lines(self, tree) -> list[tuple[int, int, str]]:
        return [(i << l, 1 << l, flat(self.text((l, i), tree))) for l, i in self.parts]

    def render(self, tree) -> str:
        body = "\n".join(f"{start}+{n}|{text}" for start, n, text in self.lines(tree))
        return f"<chat>\n{body}\n</chat>" if body else "<chat>\n</chat>"
