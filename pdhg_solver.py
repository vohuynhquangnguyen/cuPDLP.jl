"""
Vanilla PDHG (Chambolle-Pock) implementation in Python.

Given a problem in canonical form saved as .npz with keys:
  A_row, A_col, A_data, A_shape, b, c, lb, ub
Solves:
  minimize    c^T x
  subject to  A x = b,  lb <= x <= ub

No preconditioning or heuristics—just the basic PDHG loop.

Dependencies:
    numpy, scipy
"""
import numpy as np
import scipy.sparse as sp


def load_canonical_npz(npz_path: str) -> dict:
    """
    Load canonical problem arrays from an .npz file.
    Returns a dict with A_row, A_col, A_data, A_shape, b, c, lb, ub.
    """
    data = np.load(npz_path)
    return {
        'A_row': data['A_row'],
        'A_col': data['A_col'],
        'A_data': data['A_data'],
        'A_shape': tuple(data['A_shape']),
        'b': data['b'],
        'c': data['c'],
        'lb': data['lb'],
        'ub': data['ub'],
    }


def pdhg_solve(data: dict,
               max_iter: int = 1000000,
               tol: float = 1e-8,
               theta: float = 1.0) -> np.ndarray:
    """
    Basic PDHG solver (no preconditioning) for
        min_x c^T x + I_{lb<=x<=ub}(x) + I_{Ax=b}(x)

    Args:
        data: output of load_canonical_npz
        max_iter: maximum iterations
        tol: relative change tolerance for stopping
        theta: over-relaxation parameter (usually 1.0)

    Returns:
        x: solution vector
    """
    # Unpack
    A = sp.coo_matrix((data['A_data'], (data['A_row'], data['A_col'])),
                      shape=data['A_shape']).tocsr()
    AT = A.transpose()
    b = data['b']
    c = data['c']
    lb = data['lb']
    ub = data['ub']
    m, n = data['A_shape']

    # Estimate operator norm ||A|| via power iteration
    x_rand = np.random.randn(n)
    for _ in range(10):
        x_rand = AT.dot(A.dot(x_rand))
        norm_x = np.linalg.norm(x_rand)
        if norm_x > 0:
            x_rand /= norm_x
    L = np.sqrt(norm_x)
    # Step sizes satisfying tau*sigma*L^2 < 1
    sigma = 1.0 / L
    tau = 1.0 / L

    # Initialize
    x = np.zeros(n)
    x_bar = x.copy()
    y = np.zeros(m)

    for k in range(max_iter):
        # Dual ascent step (no dual prox needed for equality)
        y += sigma * (A.dot(x_bar) - b)

        # Primal gradient step
        x_old = x.copy()
        grad = AT.dot(y) + c
        x -= tau * grad
        # Proximal (box constraints)
        x = np.minimum(np.maximum(x, lb), ub)

        # Over-relaxation
        x_bar = x + theta * (x - x_old)

        # Check convergence (relative change)
        if np.linalg.norm(x - x_old) <= tol * max(1.0, np.linalg.norm(x_old)):
            print(f"PDHG converged in {k+1} iterations")
            break
    else:
        print(f"PDHG reached max_iter={max_iter} without full convergence")

    return x


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Vanilla PDHG solver")
    p.add_argument("--npz", required=True, help="Path to .npz with canonical form")
    p.add_argument("--max_iter", type=int, default=1000000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--theta", type=float, default=1.0, help="Over-relaxation parameter")
    args = p.parse_args()

    # Load problem
    data = load_canonical_npz(args.npz)

    # Solve
    x = pdhg_solve(data, max_iter=args.max_iter, tol=args.tol, theta=args.theta)

    # Compute and display objective
    optimal_value = data['c'].dot(x)
    print(f"Optimal objective value: {optimal_value}")

    # Display and save solution
    print("Solution x:", x)
    np.save("pdhg_solution.npy", x)
