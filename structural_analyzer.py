"""Structural Analyzer — preliminary EC2 reinforced-concrete design toolkit.

A Python starting point derived from the supplied MATLAB/Octave
``structural_analyzer_gui62.m``. It provides a tested Hermite beam solver,
continuous-beam load combinations, 2D truss analysis, and matrix utilities.
Run ``python structural_analyzer.py --selftest`` for headless checks.

This educational software is not a substitute for independent design review.
It does not implement every EC2 clause, detailing rule, National Annex choice,
or all five MATLAB design modules. Do not use its results for construction
without verification by a qualified structural engineer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EC2Material:
    """Basic UK-NA material parameters; stresses in MPa."""
    fck: float
    fyk: float
    alpha_cc: float = 0.85
    gamma_c: float = 1.5
    gamma_s: float = 1.15

    def __post_init__(self) -> None:
        if not 12 <= self.fck <= 50:
            raise ValueError("fck must be in [12, 50] MPa for these formulas")
        if not 250 <= self.fyk <= 600:
            raise ValueError("fyk must be in [250, 600] MPa")

    @property
    def fcd(self) -> float:
        return self.alpha_cc * self.fck / self.gamma_c

    @property
    def fyd(self) -> float:
        return self.fyk / self.gamma_s

    @property
    def fctm(self) -> float:
        return 0.30 * self.fck ** (2 / 3)

    @property
    def ecm_mpa(self) -> float:
        return 22 * ((self.fck + 8) / 10) ** 0.3 * 1000


@dataclass
class BeamResult:
    spans: np.ndarray
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    deflection: np.ndarray
    displacement_dofs: np.ndarray
    reactions: np.ndarray
    stiffness: np.ndarray


def _consistent_loads(loads: Iterable[Sequence[float]], length: float,
                      points: int = 200) -> np.ndarray:
    """Return [V1,M1,V2,M2] equivalent nodal loads.

    Load rows are (kind, magnitude, position, peak_position), where kind is
    1 point force (downward positive), 2 point moment, 3 UDL, 4 UVL.
    Point loads/moments are integrated using Hermite shape functions; the
    point moment is positive in the convention of the original MATLAB tool.
    """
    f = np.zeros(4)
    for row in loads:
        kind, value, pos, peak = row
        if kind in (1, 2) and not 0 <= pos <= length:
            raise ValueError("point load position must lie within its span")
        if kind == 1:
            r = pos / length
            shape = np.array([1 - 3*r*r + 2*r**3,
                              length*(r - 2*r*r + r**3),
                              3*r*r - 2*r**3,
                              length*(-r*r + r**3)])
            f += value * shape
        elif kind == 2:
            r = pos / length
            deriv = np.array([(-6*r + 6*r*r)/length,
                              1 - 4*r + 3*r*r,
                              (6*r - 6*r*r)/length,
                              -2*r + 3*r*r])
            f += value * deriv
        elif kind in (3, 4):
            xx = np.linspace(0, length, points + 1)
            if kind == 3:
                ww = np.full_like(xx, value)
            else:
                if not 0 <= peak <= length:
                    raise ValueError("UVL peak position must lie within its span")
                if peak == 0:
                    ww = value * (1 - xx / length)
                elif peak == length:
                    ww = value * xx / length
                else:
                    ww = np.where(xx <= peak, value * xx / peak,
                                  value * (length - xx) / (length - peak))
            r = xx / length
            shape = np.vstack((1 - 3*r**2 + 2*r**3,
                               length*(r - 2*r**2 + r**3),
                               3*r**2 - 2*r**3,
                               length*(-r**2 + r**3)))
            f += np.trapz(shape * ww, xx, axis=1)
        else:
            raise ValueError(f"unknown load kind {kind}; use 1, 2, 3, or 4")
    return f


def analyze_beam(spans: Sequence[float], ei: Sequence[float] | float,
                 loads: Iterable[Sequence[float]], left: str = "pinned",
                 right: str = "pinned", points: int = 101) -> BeamResult:
    """Analyze a continuous beam with vertical supports at each span node.

    SI-compatible units are required: lengths m, EI kN m², forces kN,
    moments kN m. Supports at the ends may be fixed, pinned, or roller;
    intermediate supports restrain vertical displacement and allow rotation.
    """
    Ls = np.asarray(spans, dtype=float).reshape(-1)
    if not len(Ls) or np.any(~np.isfinite(Ls)) or np.any(Ls <= 0):
        raise ValueError("spans must be finite positive lengths")
    EIs = np.asarray(ei, dtype=float).reshape(-1)
    if EIs.size == 1:
        EIs = np.repeat(EIs, Ls.size)
    if EIs.size != Ls.size or np.any(~np.isfinite(EIs)) or np.any(EIs <= 0):
        raise ValueError("EI must be positive and scalar or one value per span")
    if points < 3:
        raise ValueError("points must be at least 3")
    left, right = left.lower(), right.lower()
    if left not in {"fixed", "pinned", "roller"} or right not in {"fixed", "pinned", "roller"}:
        raise ValueError("end supports must be fixed, pinned, or roller")
    load_rows = list(loads)
    nnode = Ls.size + 1
    K = np.zeros((2*nnode, 2*nnode))
    rhs = np.zeros(2*nnode)
    element_data = []
    cursor = 0
    for e, (length, rigidity) in enumerate(zip(Ls, EIs), start=1):
        ke = rigidity / length**3 * np.array([
            [12, 6*length, -12, 6*length],
            [6*length, 4*length**2, -6*length, 2*length**2],
            [-12, -6*length, 12, -6*length],
            [6*length, 2*length**2, -6*length, 4*length**2]])
        dof = np.array([2*(e-1), 2*(e-1)+1, 2*e, 2*e+1])
        eloads = [(kind, val, pos, peak) for span, kind, val, pos, peak in load_rows
                  if int(span) == e]
        fe = _consistent_loads(eloads, length)
        K[np.ix_(dof, dof)] += ke
        rhs[dof] -= fe
        element_data.append((dof, ke, fe, length, rigidity, eloads, cursor))
        cursor += points
    fixed = {2*i for i in range(nnode)}  # vertical supports at all nodes
    if left == "fixed":
        fixed.add(1)
    if right == "fixed":
        fixed.add(2*(nnode-1)+1)
    free = np.array(sorted(set(range(2*nnode)) - fixed), dtype=int)
    u = np.zeros(2*nnode)
    if free.size:
        kff = K[np.ix_(free, free)]
        if np.linalg.cond(kff) > 1e14:
            raise ValueError("singular beam stiffness matrix; check supports and EI")
        u[free] = np.linalg.solve(kff, rhs[free])
    reaction = K @ u - rhs

    xs, vs, ms, ds = [], [], [], []
    node_x = np.r_[0.0, np.cumsum(Ls)]
    for dof, ke, fe, length, rigidity, eloads, _ in element_data:
        ue = ke @ u[dof] + fe
        local_x = np.linspace(0, length, points)
        v = ue[0] - ue[1]*local_x
        m = -ue[1] + ue[0]*local_x
        for kind, value, pos, peak in eloads:
            if kind == 1:
                mask = local_x >= pos
                v[mask] -= value
                m[mask] -= value * (local_x[mask] - pos)
            elif kind == 2:
                m[local_x >= pos] += value
            elif kind in (3, 4):
                if kind == 3:
                    resultant = value * local_x
                    first_moment = value * local_x**2 / 2
                else:
                    # Numerical integration of the linearly varying intensity.
                    grid = np.linspace(0, length, 801)
                    if peak == 0:
                        w = value * (1-grid/length)
                    elif peak == length:
                        w = value * grid/length
                    else:
                        w = np.where(grid <= peak, value*grid/peak,
                                     value*(length-grid)/(length-peak))
                    dx = grid[1]-grid[0]
                    accum = np.r_[0.0, np.cumsum((w[:-1]+w[1:])*0.5*dx)]
                    first = np.r_[0.0, np.cumsum((grid[:-1]*w[:-1]+grid[1:]*w[1:])*0.5*dx)]
                    resultant = np.interp(local_x, grid, accum)
                    first_moment = np.interp(local_x, grid, first)
                v -= resultant
                m -= local_x*resultant-first_moment
        # Integrate curvature and enforce the second nodal displacement.
        theta = u[dof[1]] + np.r_[0.0, np.cumsum((m[:-1]/rigidity + m[1:]/rigidity)*0.5*np.diff(local_x))]
        y = u[dof[0]] + np.r_[0.0, np.cumsum((theta[:-1]+theta[1:])*0.5*np.diff(local_x))]
        y += (u[dof[2]] - y[-1]) * local_x / length
        global_x = node_x[len(xs)//points] + local_x if xs else local_x
        xs.extend(global_x.tolist()); vs.extend(v.tolist()); ms.extend(m.tolist()); ds.extend(y.tolist())
    return BeamResult(Ls, np.asarray(xs), np.asarray(vs), np.asarray(ms),
                      np.asarray(ds), u, reaction, K)


def continuous_beam_envelopes(spans: Sequence[float], gk: Sequence[float] | float,
                              qk: Sequence[float] | float, ei: float,
                              left: str = "pinned", right: str = "pinned",
                              points: int = 101, psi2: float = 0.3) -> dict:
    """Evaluate full, alternate, and adjacent ULS patterns for UDL beam loads."""
    L = np.asarray(spans, dtype=float).reshape(-1)
    G = np.broadcast_to(np.asarray(gk, dtype=float), L.shape)
    Q = np.broadcast_to(np.asarray(qk, dtype=float), L.shape)
    patterns = [np.ones(L.size)]
    if L.size > 1:
        patterns.extend([np.arange(L.size) % 2 == 0, np.arange(L.size) % 2 == 1])
    patterns.extend(np.array([1.0 if j in (i, i+1) else 0.0
                              for j in range(L.size)])
                    for i in range(L.size-1))
    results = []
    for pattern in patterns:
        rowloads = []
        for i in range(L.size):
            w = 1.35*G[i] + 1.5*Q[i] if pattern[i] else G[i]
            rowloads.append((i+1, 3, float(w), 0.0, 0.0))
        results.append(analyze_beam(L, ei, rowloads, left, right, points))
    x = results[0].x
    # grids concatenate per span; all cases use the same coordinate stations.
    return {"x": x, "moment_max": np.max([r.moment for r in results], axis=0),
            "moment_min": np.min([r.moment for r in results], axis=0),
            "shear_max": np.max([r.shear for r in results], axis=0),
            "shear_min": np.min([r.shear for r in results], axis=0),
            "cases": results, "patterns": patterns,
            "characteristic": analyze_beam(L, ei,
                [(i+1, 3, float(G[i]+Q[i]), 0, 0) for i in range(L.size)],
                left, right, points),
            "quasi_permanent": analyze_beam(L, ei,
                [(i+1, 3, float(G[i]+psi2*Q[i]), 0, 0) for i in range(L.size)],
                left, right, points)}


def solve_truss(nodes: Sequence[Sequence[float]], members: Sequence[Sequence[float]],
                restrained: Sequence[Sequence[bool]], loads: Sequence[Sequence[float]]) -> dict:
    """Linear elastic 2D truss solver. E, A and forces use consistent units.

    Members are (node_i, node_j, E, A), with zero-based node indices.
    """
    xy = np.asarray(nodes, float)
    mem = np.asarray(members, float)
    support = np.asarray(restrained, bool)
    force = np.asarray(loads, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or support.shape != xy.shape or force.shape != xy.shape:
        raise ValueError("nodes, restrained, and loads must have shape (n, 2)")
    if mem.ndim != 2 or mem.shape[1] != 4:
        raise ValueError("members must have rows [node_i, node_j, E, A]")
    ndof = 2*len(xy); K = np.zeros((ndof, ndof)); lengths=[]; dirs=[]
    for i, row in enumerate(mem):
        ni, nj = int(row[0]), int(row[1]); E, A = row[2:]
        if ni < 0 or nj < 0 or ni >= len(xy) or nj >= len(xy) or E <= 0 or A <= 0:
            raise ValueError(f"invalid truss member {i}")
        delta = xy[nj]-xy[ni]; length = np.linalg.norm(delta)
        if length <= 0: raise ValueError(f"member {i} has zero length")
        c, s = delta/length; d = np.array([c,s]); k = E*A/length*np.outer([c,s,-c,-s],[c,s,-c,-s])
        dof=[2*ni,2*ni+1,2*nj,2*nj+1]; K[np.ix_(dof,dof)] += k
        lengths.append(length); dirs.append(d)
    F=force.reshape(-1); fixed=np.flatnonzero(support.reshape(-1)); free=np.setdiff1d(np.arange(ndof),fixed)
    u=np.zeros(ndof)
    if free.size:
        kff=K[np.ix_(free,free)]
        if np.linalg.cond(kff)>1e14: raise ValueError("singular truss stiffness matrix")
        u[free]=np.linalg.solve(kff,F[free])
    R=K@u-F; axial=[]
    for row,L,d in zip(mem,lengths,dirs):
        ni,nj=map(int,row[:2]); extension=np.dot(u[2*nj:2*nj+2]-u[2*ni:2*ni+2],d)
        axial.append(row[2]*row[3]/L*extension) # tension positive
    return {"displacements":u.reshape(-1,2), "reactions":R.reshape(-1,2),
            "member_forces":np.asarray(axial), "lengths":np.asarray(lengths), "stiffness":K}


def rref(matrix: Sequence[Sequence[float]], tol: float | None = None) -> tuple[np.ndarray, list[int]]:
    """Gauss-Jordan reduced row echelon form with partial pivoting."""
    a=np.asarray(matrix,dtype=float).copy()
    if a.ndim != 2 or not np.all(np.isfinite(a)): raise ValueError("matrix must be finite and 2D")
    threshold=tol if tol is not None else 1e-10*max(1.0,np.max(np.abs(a),initial=0))
    row=0; pivots=[]
    for col in range(a.shape[1]):
        if row >= a.shape[0]: break
        p=row+int(np.argmax(np.abs(a[row:,col])))
        if abs(a[p,col]) <= threshold: continue
        a[[row,p]]=a[[p,row]]; a[row]/=a[row,col]
        for other in range(a.shape[0]):
            if other != row: a[other]-=a[other,col]*a[row]
        pivots.append(col); row+=1
    a[np.abs(a)<threshold]=0
    return a,pivots


def matrix_toolbox(operation: str, a: Sequence[Sequence[float]], b=None):
    """Small matrix toolbox: multiply, rref, inverse, solve, determinant, rank."""
    A=np.asarray(a,dtype=float)
    if operation == "multiply": return A@np.asarray(b,dtype=float)
    if operation == "rref": return rref(A)
    if operation == "inverse":
        if A.shape[0]!=A.shape[1]: raise ValueError("inverse requires square matrix")
        R,piv=rref(np.c_[A,np.eye(len(A))])
        if any(p>=len(A) for p in piv) or len([p for p in piv if p<len(A)])<len(A):
            raise ValueError("matrix is singular")
        return R[:,len(A):]
    if operation == "solve": return np.linalg.solve(A,np.asarray(b,dtype=float))
    if operation == "determinant": return float(np.linalg.det(A))
    if operation == "rank": return int(np.linalg.matrix_rank(A))
    raise ValueError("operation must be multiply, rref, inverse, solve, determinant, or rank")


def selftest() -> bool:
    """Run small analytical sanity checks, without a display server."""
    failures=[]
    # Simply supported 5 m beam, 10 kN/m: Mmax=wL²/8, delta=5wL⁴/(384EI).
    result=analyze_beam([5],10000,[(1,3,10,0,0)],"pinned","roller",401)
    expected_m=10*25/8
    expected_d=5*10*5**4/(384*10000)
    if abs(np.max(result.moment)-expected_m)>0.05: failures.append("simply supported beam moment")
    if abs(np.max(np.abs(result.deflection))-expected_d)>2e-4: failures.append("simply supported beam deflection")
    # Truss equilibrium and matrix examples.
    tr=solve_truss([[0,0],[4,0],[2,3]],[[0,1,200e6,1e-3],[0,2,200e6,1e-3],[1,2,200e6,1e-3]],
                   [[1,1],[0,1],[0,0]],[[0,0],[0,0],[0,-10]])
    if abs(tr["reactions"][:,1].sum()-10)>1e-7: failures.append("truss vertical equilibrium")
    rr,piv=rref([[2,1,-1,8],[-3,-1,2,-11],[-2,1,2,-3]])
    if len(piv)!=3 or not np.allclose(rr[:,-1],[2,3,-1]): failures.append("Gauss-Jordan example")
    mat=EC2Material(30,500)
    if not np.isclose(mat.fcd,17.0) or not np.isclose(mat.fyd,500/1.15): failures.append("EC2 material values")
    if failures:
        print("SELFTEST FAILED: "+", ".join(failures)); return False
    print("SELFTEST PASSED: beam, truss, matrix toolbox, EC2 material primitives")
    return True


def main() -> int:
    parser=argparse.ArgumentParser(description="Preliminary structural analysis toolkit")
    parser.add_argument("--selftest",action="store_true",help="run headless sanity checks")
    args=parser.parse_args()
    if args.selftest: return 0 if selftest() else 1
    parser.print_help()
    print("Import analyze_beam, continuous_beam_envelopes, solve_truss, or matrix_toolbox from this module.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
