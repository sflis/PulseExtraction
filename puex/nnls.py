import numpy as np
from numba import jit


@jit(nopython=True)
def nnls(A, b, maxit=None, tolerance=0):
    """A python implementation of the Lawson/Hanson active set algorithm



    See:
        "A Comparison of Block Pivoting and Interior-Point Algorithms for Linear Least
        Squares Problems with Nonnegative Variables"
        Author(s): Luis F. Portugal, Joaquim J. Judice, Luis N. Vicente
        Source: Mathematics of Computation, Vol. 63, No. 208 (Oct., 1994), pp. 625-643
        Published by: American Mathematical Society
        Stable URL: http://www.jstor.org/stable/2153286

    Args:
        A (): Description
        b (TYPE): Description
        maxit (None, optional): Description
        tolerance (int, optional): Description

    Returns:
        TYPE: Description
    """

    # step 0
    F = []  # passive (solved for) set
    G = list(range(A.shape[1]))  # active (clamped to zero) set
    x = np.zeros(A.shape[1])

    y = -np.dot(A.transpose(), b)
    if maxit is None:
        maxit = len(b)

    iterations = 0
    lstsqs = 0

    while True:
        if iterations >= maxit:
            break
        iterations += 1
        # step 1
        if len(G) == 0:
            break  # the active set is the whole set, we're done
        r_G = y[np.array(G)].argmin()
        r = G[r_G]

        if y[r] >= tolerance:
            break  # x is the optimal solution, we're done
        F.append(r)
        F.sort()
        G.remove(r)
        feasible = False
        while not feasible:
            # step 2
            x_F = np.linalg.lstsq(A[:, np.array(F)], b)[0]
            lstsqs += 1
            if (x_F >= 0).all():
                x[np.array(F)] = x_F
                feasible = True
            else:
                # if the new trial solution gained a negative element
                mask = x_F <= 0
                theta = x[np.array(F)] / (x[np.array(F)] - x_F)

                r_F = theta[mask].argmin()
                alpha = theta[mask][r_F]
                r = np.array(F)[mask][r_F]
                x[np.array(F)] = x[np.array(F)] + alpha * (x_F - x[np.array(F)])
                F.remove(r)
                G.append(r)
                G.sort()
        # step 3
        y[:] = 0
        y[np.array(G)] = np.dot(
            A[:, np.array(G)].transpose(),
            (np.dot(A[:, np.array(F)], x[np.array(F)]) - b),
        )
    #
    return x


@jit(nopython=True, cache=True)
def nnls2(A, b, maxit=None, tolerance=0, lamb1=0, lamb2=0, n=1):
    """A mockup of the Lawson/Hanson active set algorithm

    See:
    "A Comparison of Block Pivoting and Interior-Point Algorithms for Linear Least
     Squares Problems with Nonnegative Variables"
    Author(s): Luis F. Portugal, Joaquim J. Judice, Luis N. Vicente
    Source: Mathematics of Computation, Vol. 63, No. 208 (Oct., 1994), pp. 625-643
    Published by: American Mathematical Society
    Stable URL: http://www.jstor.org/stable/2153286
    """

    # step 0
    F = []  # passive (solved for) set
    G = list(range(A.shape[1]))  # active (clamped to zero) set
    x = np.zeros(A.shape[1])

    y = -np.dot(A.transpose(), b)
    if maxit is None:
        maxit = len(b)

    iterations = 0
    lstsqs = 0
    x_ret = []
    L_ret = []
    r_last = None
    r_G = y[np.array(G)].argmin()
    r = G[r_G]
    L_ret.append(y.copy())
    x_ret.append(np.zeros(len(y)))
    while True:
        if iterations >= maxit:
            break
        iterations += 1
        # step 1
        if len(G) == 0:
            break  # the active set is the whole set, we're done
        r_G = -1
        Ga = np.array(G)
        Fa = np.array(F)
        # NFC modification
        if len(F) > n:
            iF = np.where(r_last == Fa)[0][0]
            for i in [iF - 1, iF]:

                if (i < 0) or (i + 1) >= Fa.shape[0]:
                    continue
                m = (Fa[i] < Ga) & (Fa[i + 1] > Ga)
                if (np.sum(m) > 0) and (y[Ga[m]] < 0).all():
                    r_G = y[Ga[m]].argmin()
                    r = Ga[m][np.uint32(r_G)]
                    r_last = r
                    break

        if r_G == -1:  # NNLS Default
            r_G = y[Ga].argmin()
            r = Ga[r_G]
            r_last = r

            if y[r] >= tolerance:
                found = False
                for iF in range(Fa.shape[0]):
                    for i in [iF - 1, iF]:
                        if (i < 0) or (i + 1) >= Fa.shape[0]:
                            continue
                        m = (Fa[i] < Ga) & (Fa[i + 1] > Ga)
                        if (np.sum(m) > 0) and (y[Ga[m]] < 0).all():
                            r_G = y[Ga[m]].argmin()
                            r = Ga[m][np.uint64(r_G)]
                            r_last = r
                            if y[r] >= -0.3:
                                continue
                            found = True
                            break
                if not found or len(F) < n:
                    break  # x is the optimal solution, we're done

        F.append(r)
        F.sort()
        G.remove(r)
        feasible = False
        while not feasible:
            # step 2
            x_F = np.linalg.lstsq(A[:, np.array(F)], b)[0]
            lstsqs += 1
            if (x_F >= 0).all():
                x[np.array(F)] = x_F
                feasible = True
            else:
                # if the new trial solution gained a negative element
                mask = x_F <= 0
                theta = x[np.array(F)] / (x[np.array(F)] - x_F)

                r_F = theta[mask].argmin()
                alpha = theta[mask][r_F]
                r = np.array(F)[mask][r_F]
                x[np.array(F)] = x[np.array(F)] + alpha * (x_F - x[np.array(F)])
                x[
                    r
                ] = 0  # I had to add this as sometimes basis in the active set still had >0 coefficients
                F.remove(r)
                G.append(r)
                G.sort()
        # step 3
        Ga = np.array(G)
        Fa = np.array(F)
        y[:] = 0
        xv = x[Fa]
        L = (
            (np.dot(A[:, Fa], xv) - b)
            + lamb1 * np.linalg.norm(xv)
            + lamb2 * np.dot(xv, xv)
        )
        y[Ga] = np.dot(A[:, Ga].transpose(), L)
        x_ret.append(x.copy())
        L_ret.append(y.copy())

    return x_ret, L_ret


@jit(nopython=True, cache=True)
def nnlsfixed_order(A, b, GG, maxit=None, tolerance=0, lamb1=0.1, lamb2=0.1):
    """A mockup of the Lawson/Hanson active set algorithm

    See:
    "A Comparison of Block Pivoting and Interior-Point Algorithms for Linear
    Least Squares Problems with Nonnegative Variables"
    Author(s): Luis F. Portugal, Joaquim J. Judice, Luis N. Vicente
    Source: Mathematics of Computation, Vol. 63, No. 208 (Oct., 1994), pp. 625-643
    Published by: American Mathematical Society
    Stable URL: http://www.jstor.org/stable/2153286
    """

    # step 0
    F = []  # passive (solved for) set
    G = list(range(A.shape[1]))  # active (clamped to zero) set
    x = np.zeros(A.shape[1])

    y = -np.dot(A.transpose(), b)
    if maxit is None:
        maxit = len(b)

    iterations = 0
    lstsqs = 0
    x_ret = []
    L_ret = []
    # r_last = None
    r_G = y[np.array(G)].argmin()
    r = G[r_G]
    L_ret.append(y.copy())
    x_ret.append(np.zeros(len(y)))
    while True:
        if iterations >= maxit:
            break
        iterations += 1
        # step 1
        if len(G) == 0:
            break  # the active set is the whole set, we're done
        if iterations < 4:
            r = GG.pop()  # G[r_G]
        else:
            r_G = y[np.array(G)].argmin()
            r = G[r_G]

        if y[r] >= tolerance:
            break  # x is the optimal solution, we're done

        F.append(r)
        F.sort()
        G.remove(r)
        feasible = False
        while not feasible:
            # step 2
            x_F = np.linalg.lstsq(A[:, np.array(F)], b)[0]
            lstsqs += 1
            if (x_F >= 0).all():
                x[np.array(F)] = x_F
                feasible = True
            else:
                # if the new trial solution gained a negative element
                mask = x_F <= 0
                theta = x[np.array(F)] / (x[np.array(F)] - x_F)

                r_F = theta[mask].argmin()
                alpha = theta[mask][r_F]
                r = np.array(F)[mask][r_F]
                x[np.array(F)] = x[np.array(F)] + alpha * (x_F - x[np.array(F)])

                F.remove(r)
                G.append(r)
                G.sort()
        # step 3
        y[:] = 0
        xv = x[np.array(F)]
        LL = np.dot(A[:, np.array(F)], xv) - b
        L = (
            LL
            + lamb1 * np.linalg.norm(LL)  # np.linalg.norm(xv)
            + lamb2 * np.dot(xv, xv)
        )
        y[np.array(G)] = np.dot(A[:, np.array(G)].transpose(), L)
        x_ret.append(x.copy())
        L_ret.append(y.copy())
    return x_ret, L_ret


@jit(nopython=True)
def nnls_sortedbackfeed(model, waveform, tol):
    params, L = nnls2(
        model, waveform, tolerance=0, lamb1=+0.0, lamb2=0.0, n=1000
    )  # Full NNLS for charge ordered NNLS
    c = params[-1]
    G = np.array(sorted(zip(c, np.arange(len(c))), key=lambda x: x[0]), dtype=np.int64)[
        :, 1
    ]
    return nnlsfixed_order(
        model, waveform, list(G), tolerance=tol, lamb1=+0.0, lamb2=0.0
    )
