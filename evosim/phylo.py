"""Reconstrucción filogenética desde secuencias: distancias, Neighbor-Joining y comparación
de topologías (Robinson-Foulds)."""
import numpy as np

from .stats import jukes_cantor


def p_distance_matrix(seqs):
    """seqs: (n, L). Proporción de sitios distintos entre cada par."""
    seqs = np.asarray(seqs)
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = (seqs != seqs[i]).mean(axis=1)
    return D


def jc_distance_matrix(seqs):
    """Distancias corregidas por sustituciones múltiples (Jukes & Cantor 1969)."""
    return jukes_cantor(p_distance_matrix(seqs))


def mean_pairwise_distance(ga, gb, sites=None, max_pairs=2000, rng=None):
    """d_xy: diferencias medias por sitio entre haplotipos de dos poblaciones (Nei & Li 1979)."""
    rng = rng or np.random.default_rng(0)
    ha, hb = ga.reshape(-1, ga.shape[-1]), gb.reshape(-1, gb.shape[-1])
    if sites is not None:
        ha, hb = ha[:, sites], hb[:, sites]
    i = rng.integers(0, len(ha), max_pairs)
    j = rng.integers(0, len(hb), max_pairs)
    return float((ha[i] != hb[j]).mean())


class Node:
    def __init__(self, name=None, children=(), length=0.0):
        self.name, self.children, self.length = name, list(children), length

    def leaves(self):
        if not self.children:
            return [self.name]
        return [x for c in self.children for x in c.leaves()]

    def newick(self, digits=4):
        if not self.children:
            s = str(self.name)
        else:
            s = "(" + ",".join(c.newick(digits) for c in self.children) + ")"
        return s + f":{self.length:.{digits}g}"


def neighbor_joining(D, names):
    """Neighbor-Joining (Saitou & Nei 1987). Devuelve un árbol sin raíz (trifurcación en la base)."""
    D = np.array(D, dtype=float)
    nodes = [Node(n) for n in names]
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        li = 0.5 * D[i, j] + (r[i] - r[j]) / (2 * (n - 2))
        lj = D[i, j] - li
        nodes[i].length, nodes[j].length = max(li, 0.0), max(lj, 0.0)
        new = Node(children=[nodes[i], nodes[j]])
        d_new = 0.5 * (D[i] + D[j] - D[i, j])
        keep = [k for k in range(n) if k not in (i, j)]
        D = np.vstack([np.c_[D[np.ix_(keep, keep)], d_new[keep]], np.r_[d_new[keep], 0.0]])
        nodes = [nodes[k] for k in keep] + [new]
    if len(nodes) == 3:
        a, b, c = nodes
        dab, dac, dbc = D[0, 1], D[0, 2], D[1, 2]
        a.length = max(0.5 * (dab + dac - dbc), 0.0)
        b.length = max(0.5 * (dab + dbc - dac), 0.0)
        c.length = max(0.5 * (dac + dbc - dab), 0.0)
    return Node(children=nodes)


def parse_newick(s):
    """Parser mínimo de Newick (nombres y longitudes)."""
    s = s.strip().rstrip(";")
    pos = 0

    def node():
        nonlocal pos
        children = []
        if s[pos] == "(":
            pos += 1
            children.append(node())
            while s[pos] == ",":
                pos += 1
                children.append(node())
            pos += 1                      # ')'
        start = pos
        while pos < len(s) and s[pos] not in ",():":
            pos += 1
        name = s[start:pos] or None
        length = 0.0
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in ",()":
                pos += 1
            length = float(s[start:pos])
        return Node(name, children, length)

    return node()


def splits(tree):
    """Biparticiones no triviales del árbol (como conjuntos del lado sin la primera hoja)."""
    leaves = sorted(tree.leaves())
    ref = leaves[0]
    allset = frozenset(leaves)
    out = set()

    def rec(node):
        if not node.children:
            return frozenset([node.name])
        s = frozenset().union(*(rec(c) for c in node.children))
        if 1 < len(s) < len(allset) - 1:
            out.add(s if ref not in s else allset - s)
        return s

    rec(tree)
    return out


def robinson_foulds(t1, t2):
    """Distancia de Robinson-Foulds: biparticiones presentes en un árbol y no en el otro."""
    if sorted(t1.leaves()) != sorted(t2.leaves()):
        raise ValueError("los árboles tienen hojas distintas")
    a, b = splits(t1), splits(t2)
    return len(a ^ b)


def collapse_unary(tree):
    """Quita nodos con un solo hijo (p. ej. linajes extintos)."""
    kids = [collapse_unary(c) for c in tree.children]
    if len(kids) == 1:
        k = kids[0]
        k.length += tree.length
        return k
    return Node(tree.name, kids, tree.length)
