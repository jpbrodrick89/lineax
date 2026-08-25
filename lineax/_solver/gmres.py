# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
from collections.abc import Callable
from typing import Any, cast, TypeAlias

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Float, Inexact, PyTree

from .._misc import structure_equal
from .._norm import max_norm, two_norm
from .._operator import AbstractLinearOperator, conj, linearise
from .._solution import RESULTS
from .base import AbstractLinearSolver
from .misc import preconditioner_and_y0


_GMRESState: TypeAlias = AbstractLinearOperator


class GMRES(AbstractLinearSolver[_GMRESState]):
    """GMRES solver for linear systems.

    The operator should be square.

    Similar to `jax.scipy.sparse.linalg.gmres`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][]. This method uses left preconditioning,
        so it is the preconditioned residual that is minimized, though the actual
        termination criteria uses the un-preconditioned residual.
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    """

    rtol: float
    atol: float
    norm: Callable = max_norm
    max_steps: int | None = None
    restart: int = 20
    stagnation_iters: int = 20

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")

        if isinstance(self.atol, (int, float)) and isinstance(self.rtol, (int, float)):
            if self.atol == 0 and self.rtol == 0 and self.max_steps is None:
                raise ValueError(
                    "Must specify `rtol`, `atol`, or `max_steps` (or some combination "
                    "of all three)."
                )

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if not structure_equal(operator.in_structure(), operator.out_structure()):
            raise ValueError(
                "`GMRES(..., normal=False)` may only be used for linear solves with "
                "square matrices."
            )
        return linearise(operator)

    #
    # This differs from `jax.scipy.sparse.linalg.gmres` in a few ways:
    # 1. We use a more sophisticated termination condition. To begin with we have an
    #    rtol and atol in the conventional way, inducing a vector-valued scale. This is
    #    then checked in both the `y` and `b` domains (for `Ay = b`).
    # 2. We handle in-place updates with buffers to avoid generating unnecessary
    #    copies of arrays during the Gram-Schmidt procedure.
    # 3. We build the QR factorisation of the Hessenberg matrix incrementally via
    #    Givens rotations during the Arnoldi process itself (as in
    #    `scipy.sparse.linalg.gmres` and `jax.scipy.sparse.linalg.gmres`'s
    #    `solve_method="incremental"`), rather than solving a dense linear problem
    #    from scratch once the Krylov basis is complete. This both gives a running
    #    residual estimate for free, allowing early exit within a restart cycle once
    #    within tolerance, and avoids ever forming the (worse-conditioned) normal
    #    equations.
    # 4. We use tricks to compile `A y` fewer times throughout the code, including
    #    passing a dummy initial residual.
    # 5. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 6. We do not use the unnecessary loop within Gram-Schmidt, and simply compute
    #    this in a single pass.
    # 7. We add better safety checks for breakdown, and a safety check for stagnation
    #    of the iterates even when we don't explicitly get breakdown.
    #
    def compute(
        self,
        state: _GMRESState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        operator = state
        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        if has_scale:
            b_norm = self.norm(vector)
            b_scale = self.atol + self.rtol * b_norm
            # Inner (preconditioned) convergence tolerance used to allow early exit
            # *within* a restart cycle -- distinct from the outer `b_scale`, since the
            # Arnoldi process runs on preconditioned residuals. Mirrors the `ptol`
            # used by `scipy.sparse.linalg.gmres` / `jax.scipy.sparse.linalg.gmres`.
            Mb_norm = self.norm(preconditioner.mv(vector))
            ptol = Mb_norm * jnp.minimum(
                1.0, jnp.where(b_norm > 0, b_scale / b_norm, 1.0)
            )
        else:
            # No tolerance was specified, so there is no meaningful notion of "early":
            # always run each restart cycle out to `restart` steps (or breakdown), as
            # before.
            ptol = -jnp.inf
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        if self.max_steps is None:
            max_steps = 10 * size  # Copied from SciPy!
        else:
            max_steps = self.max_steps
        restart = min(self.restart, size)

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            if has_scale:
                # Standard relative-residual stopping rule: ‖r‖ ≤ atol + rtol·‖b‖
                # (and likewise for the increment in the `y` space). Note this uses
                # scalar norms, *not* an elementwise `atol + rtol·|b|` scale: the
                # latter is unsatisfiable for wide-dynamic-range `b`, where the
                # round-off floor of large components exceeds the absolute tolerance
                # demanded of small ones, yielding spurious non-convergence.
                y_scale = self.atol + self.rtol * self.norm(y)
                b_unconverged = self.norm(r) > b_scale  # pyright: ignore
                y_unconverged = self.norm(diff) > y_scale
                return b_unconverged | y_unconverged
            else:
                return True

        def cond_fun(carry):
            y, r, _, deferred_breakdown, diff, _, step, stagnation_counter = carry
            # NOTE: we defer ending due to breakdown by one loop! This is nonstandard,
            # but lets us use a cauchy-like condition in the convergence criteria.
            # If we do not defer breakdown, breakdown may detect convergence when
            # the diff between two iterations is still quite large, and we only
            # consider convergence when the diff is small.
            out = jnp.invert(deferred_breakdown) & (
                stagnation_counter < self.stagnation_iters
            )
            out = out & not_converged(r, diff, y)
            out = out & (step < max_steps)
            # The first pass uses a dummy value for r0 in order to save on compiling
            # an extra matvec. The dummy step may raise a breakdown, and `step == 0`
            # avoids us from returning prematurely.
            return out | (step == 0)

        def body_fun(carry):
            # `breakdown` -> `deferred_breakdown` and `deferred_breakdown` -> `_`
            y, r, deferred_breakdown, _, diff, r_min, step, stagnation_counter = carry
            y_new, r_new, breakdown, diff_new = self._gmres_compute(
                operator, vector, y, r, restart, preconditioner, step == 0, ptol
            )

            #
            # If the minimum residual does not decrease for many iterations
            # ("many" is determined by self.stagnation_iters) then the iterative
            # solve has stagnated and we stop the loop. This bit keeps track of how
            # long it has been since the minimum has decreased, and updates the minimum
            # when a new minimum is encountered. As far as I (raderj) am
            # aware, this is custom to our implementation and not standard practice.
            #
            r_new_norm = self.norm(r_new)
            r_decreased = (r_new_norm - r_min) < 0
            stagnation_counter = jnp.where(r_decreased, 0, stagnation_counter + 1)
            stagnation_counter = cast(Array, stagnation_counter)
            r_min = jnp.minimum(r_new_norm, r_min)

            return (
                y_new,
                r_new,
                breakdown,
                deferred_breakdown,
                diff_new,
                r_min,
                step + 1,
                stagnation_counter,
            )

        # Initialise the residual r0 to the dummy value of all 0s. This means
        # the first iteration of Gram-Schmidt will do nothing, but it saves
        # us from compiling an extra matvec here.
        r0 = ω(vector).call(jnp.zeros_like).ω
        init_carry = (
            y0,  # y
            r0,  # residual
            False,  # breakdown
            False,  # deferred_breakdown
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,  # diff
            jnp.inf,  # r_min
            0,  # steps
            jnp.array(0),  # stagnation counter
        )
        (
            solution,
            residual,
            _,  # breakdown
            breakdown,  # deferred_breakdown
            diff,
            _,
            num_steps,
            stagnation_counter,
        ) = lax.while_loop(cond_fun, body_fun, init_carry)

        if self.max_steps is None:
            result = RESULTS.where(
                num_steps == max_steps, RESULTS.singular, RESULTS.successful
            )
        elif has_scale:
            result = RESULTS.where(
                num_steps == max_steps, RESULTS.max_steps_reached, RESULTS.successful
            )
        else:
            result = RESULTS.successful

        result = RESULTS.where(
            stagnation_counter >= self.stagnation_iters, RESULTS.stagnation, result
        )

        # breakdown is only an issue if we broke down outside the tolerance
        # of the solution. If we get breakdown and are within the tolerance,
        # this is called convergence :)
        breakdown = breakdown & not_converged(residual, diff, solution)
        # breakdown is the most serious potential issue
        result = RESULTS.where(breakdown, RESULTS.breakdown, result)

        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def _gmres_compute(
        self, operator, vector, y, r, restart, preconditioner, first_pass, ptol
    ):
        #
        # internal function for computing the bulk of the gmres. We seperate this out
        # for two reasons:
        # 1. avoid nested body and cond functions in the body and cond function of
        # `self.compute`. `self.compute` is primarily responsible for the restart
        # behavior of gmres.
        # 2. Like the jax.scipy implementation we may want to add an incremental
        # version at a later date.
        #

        def main_gmres(y):
            # see the comment at the end of `_arnoldi_gram_schmidt` for a discussion
            # of `initial_breakdown`
            r_normalised, r_norm, initial_breakdown = self._normalise(r, eps=None)
            basis_init = jtu.tree_map(
                lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
                r_normalised,
            )
            dtype = jnp.result_type(*jtu.tree_leaves(r_normalised))
            coeff_mat_init = jnp.eye(restart, restart + 1, dtype=dtype)
            # `givens[k]` stores the `(cs, sn)` pair of the Givens rotation that
            # eliminated the subdiagonal entry introduced by Arnoldi step `k`. Once
            # `coeff_mat` has had every past rotation (re-)applied to its new row, and
            # a fresh rotation applied to eliminate its own subdiagonal entry, it holds
            # (transposed) the upper-triangular `R` factor of the Hessenberg matrix's
            # QR factorisation -- built up incrementally, one Arnoldi step at a time,
            # rather than factorised in one shot at the end. `beta_vec` is the
            # right-hand side of the (restart+1)-dimensional least-squares problem,
            # rotated by the exact same sequence of Givens rotations; `abs(beta_vec[k
            # + 1])` is then the norm of the residual of the (size-`k` truncated)
            # least-squares problem, i.e. a running residual estimate obtained for
            # free, with no extra matrix-vector product. This lets us exit as soon as
            # that estimate is within tolerance, rather than always building the full
            # `restart`-dimensional Krylov subspace. Mirrors
            # `scipy.sparse.linalg.gmres` and `jax.scipy.sparse.linalg.gmres`'s
            # `solve_method="incremental"`.
            givens_init = jnp.zeros((restart, 2), dtype=dtype)
            beta_vec_init = (
                jnp.zeros((restart + 1,), dtype=dtype).at[0].set(r_norm.astype(dtype))
            )

            def cond_fun(carry):
                _, _, _, _, err, breakdown, step = carry
                return (step < restart) & jnp.invert(breakdown) & (err > ptol)

            def body_fun(carry):
                basis, coeff_mat, givens, beta_vec, err, breakdown, step = carry
                (
                    basis_new,
                    coeff_mat_new,
                    givens_new,
                    beta_vec_new,
                    err_new,
                    breakdown,
                ) = self._arnoldi_gram_schmidt(
                    operator,
                    preconditioner,
                    basis,
                    coeff_mat,
                    givens,
                    beta_vec,
                    err,
                    step,
                    restart,
                    vector,
                    breakdown,
                )
                return (
                    basis_new,
                    coeff_mat_new,
                    givens_new,
                    beta_vec_new,
                    err_new,
                    breakdown,
                    step + 1,
                )

            def buffers(carry):
                basis, coeff_mat, _, _, _, _, _ = carry
                return basis, coeff_mat

            init_carry = (
                basis_init,
                coeff_mat_init,
                givens_init,
                beta_vec_init,
                r_norm,  # `err`: real-valued, always the norm of a residual.
                initial_breakdown,
                0,
            )
            basis, coeff_mat, _, beta_vec, _, breakdown, steps = eqxi.while_loop(
                cond_fun, body_fun, init_carry, kind="lax", buffers=buffers
            )
            # The rotation that eliminates each new subdiagonal entry deposits the
            # *leftover* residual magnitude (what `err` tracks) at `beta_vec[steps]`.
            # When `steps == restart` this is `beta_vec[restart]`, already excluded by
            # `[:-1]` below. But when we stop earlier than that -- exiting once `err`
            # is within tolerance, rather than by exhausting `restart` steps or
            # breakdown -- that leftover sits at an index that *is* included in the
            # `[:-1]` slice, and solving the triangular system would then wrongly
            # attribute it to the (real, but not-yet-incorporated) Krylov direction
            # `steps`, corrupting `z` by an amount of the same order as `err` itself.
            # It must be zeroed out first, matching the fact that this direction has
            # deliberately not been incorporated into the approximation.
            beta_vec = beta_vec.at[steps].set(0)
            # `coeff_mat.T`'s leading `restart` rows are exactly the upper-triangular
            # `R` factor accumulated above (untouched rows -- from early exit or
            # breakdown -- retain their `coeff_mat_init` identity row, which is still
            # consistent with upper-triangularity), so a triangular solve replaces the
            # dense QR solve of the whole Hessenberg system used previously; besides
            # being cheaper, this also means we no longer need a `linear_solve` call
            # (and the circular-import workaround it required) to solve it.
            z = jsp.linalg.solve_triangular(coeff_mat[:, :-1].T, beta_vec[:-1])
            diff = jtu.tree_map(
                lambda mat: jnp.tensordot(
                    mat[..., :-1], z, axes=1, precision=lax.Precision.HIGHEST
                ),
                basis,
            )
            y_new = (y**ω + diff**ω).ω
            return y_new, diff, breakdown

        def first_gmres(y):
            return y, ω(y).call(lambda x: jnp.full_like(x, jnp.inf)).ω, False

        first_pass = eqxi.unvmap_any(first_pass)
        y_new, diff, breakdown = lax.cond(first_pass, first_gmres, main_gmres, y)
        r_new = preconditioner.mv((vector**ω - operator.mv(y_new) ** ω).ω)

        return y_new, r_new, breakdown, diff

        # NOTE: in the jax implementation:
        # https://github.com/google/jax/blob/
        # c662fd216dec10cdb2cff4138b4318bb98853134/jax/_src/scipy/sparse/linalg.py#L327
        # _classical_iterative_gram_schmidt uses a while loop to call this.
        # However, max_iterations is set to 2 in all calls they make to the function,
        # and the condition function requires steps < (max_iterations - 1).
        # This means that in fact they only apply Gram-Schmidt once, and using a
        # while_loop is unnecessary.

    def _arnoldi_gram_schmidt(
        self,
        operator,
        preconditioner,
        basis,
        coeff_mat,
        givens,
        beta_vec,
        err,
        step,
        restart,
        vector,
        initial_breakdown,
    ):
        #
        # compute `basis.T @ basis_step` for each leaf of pytree
        # and then compute the projected vector onto the basis
        #
        # `basis` is a pytree with buffers, meaning it can only be
        # indexed into. Through this section, there are terms like `lambda _, x: ...`
        # because`jtu.tree_map` only uses the first argument to determine the shape
        # of the pytree. Since _Buffer is considered part of the pytree
        # structure, we get leaves which are not buffers if we directly pass `basis`.
        # Instead, we make sure that the first argument of the tree map is something
        # with the correct pytree structure, such as `vector` in the dummy case and
        # basis_step when not, so that we correctly index into `basis`.
        #
        basis_step = preconditioner.mv(
            operator.mv(jtu.tree_map(lambda _, x: x[..., step], vector, basis))
        )
        step_norm = two_norm(basis_step)
        contract_matrix = lambda x, y: ft.partial(
            jnp.tensordot, axes=x.ndim, precision=lax.Precision.HIGHEST
        )(x, y[...].conj())
        _proj = jtu.tree_map(contract_matrix, basis_step, basis)
        proj = jtu.tree_reduce(lambda x, y: x + y, _proj)
        proj_on_cols = jtu.tree_map(lambda _, x: x[...] @ proj, vector, basis)
        # now remove the component of the vector in that subspace
        basis_step_new = (basis_step**ω - proj_on_cols**ω).ω
        eps = step_norm * jnp.finfo(proj.dtype).eps
        basis_step_normalised, step_norm_new, breakdown = self._normalise(
            basis_step_new, eps=eps
        )
        basis_new = jtu.tree_map(
            lambda y, mat: mat.at[..., step + 1].set(y),
            basis_step_normalised,
            basis,
        )
        proj_new = proj.at[step + 1].set(step_norm_new.astype(jnp.result_type(proj)))

        # Fold `proj_new` (the Hessenberg matrix's column `step`, i.e. the overlaps of
        # this step's Krylov vector with every previous one, plus its own norm at
        # `step + 1`) into the incrementally-built QR factorisation: first re-apply
        # every previously-computed rotation (each one only ever touches the pair of
        # entries it originally eliminated), then construct and apply one new
        # rotation to eliminate this column's own subdiagonal entry at `step + 1`.
        # `beta_vec`, the right-hand side of the same least-squares problem, is
        # rotated identically, so that `abs(rotated_beta_vec[step + 1])` becomes the
        # (cheap, exact) norm of the leftover residual.
        def apply_kth_rotation(k, row):
            return self._rotate_vector(row, k, givens[k, 0], givens[k, 1])

        rotated_row = lax.fori_loop(0, step, apply_kth_rotation, proj_new)
        cs, sn = self._givens_rotation(rotated_row[step], rotated_row[step + 1])
        triangular_row = self._rotate_vector(rotated_row, step, cs, sn)
        rotated_beta_vec = self._rotate_vector(beta_vec, step, cs, sn)
        err_new = jnp.abs(rotated_beta_vec[step + 1])

        #
        # NOTE: two somewhat complicated things are going on here:
        #
        # The `coeff_mat` in_place update has a batch tracer, so we need to be
        # careful and wrap it in a buffer, hence the use of eqxi.while_loop
        # instead of lax.while_loop throughout.
        #
        # `initial_breakdown` occurs when the previous loop returns a
        # residual which is small enough to be interpreted as 0 by self._normalise,
        # but which was passed through the solver anyway. This occurs when
        # the residual is small but the diff is not, or if the
        # correct solution was given to GMRES from the start. Both of these tend to
        # happen at the start of `gmres_compute`.
        # The latter may happen when using a sequence of iterative methods.
        # If `initial_breakdown` occurs, then we leave `coeff_mat`, `givens`, and
        # `beta_vec` as they were at initialisation (under `vmap`, other batch
        # elements may still be part-way through this same loop, so we select rather
        # than skip): replacing `coeff_mat`'s row with the projection (which will be
        # all 0s) would mean the triangular solve at the end divides by a zero
        # diagonal entry.
        #
        keep = initial_breakdown
        coeff_mat_new = coeff_mat.at[step, :].set(triangular_row, pred=jnp.invert(keep))
        givens_step = givens.at[step, :].set(jnp.array([cs, sn]))
        givens_new = jnp.where(keep, givens, givens_step)
        beta_vec_new = jnp.where(keep, beta_vec, rotated_beta_vec)
        err_new = jnp.where(keep, err, err_new)
        return basis_new, coeff_mat_new, givens_new, beta_vec_new, err_new, breakdown

    def _normalise(
        self, x: PyTree[Array], eps: Float[ArrayLike, ""] | None
    ) -> tuple[PyTree[Array], Inexact[Array, ""], Bool[ArrayLike, ""]]:
        norm = two_norm(x)
        if eps is None:
            eps = jnp.finfo(norm.dtype).eps
        else:
            eps = jnp.astype(eps, norm.dtype)
        breakdown = norm < eps  # pyright: ignore
        safe_norm = jnp.where(breakdown, jnp.inf, norm)
        with jax.numpy_dtype_promotion("standard"):
            x_normalised = (x**ω / safe_norm).ω
        return x_normalised, norm, breakdown

    def _givens_rotation(self, a, b):
        # Constructs `cs`, `sn` such that applying `_rotate_vector` at a pair of
        # entries holding `(a, b)` zeroes out the second of the two. Safe against
        # `a == 0`, `b == 0`, and overflow for large `|a|`/`|b|` (never squares
        # either input, unlike the naive `r = sqrt(a**2 + b**2)`).
        b_zero = jnp.abs(b) == 0
        a_lt_b = jnp.abs(a) < jnp.abs(b)
        t = -jnp.where(a_lt_b, a, b) / jnp.where(a_lt_b, b, a)
        r = lax.rsqrt(1 + jnp.abs(t) ** 2).astype(t.dtype)
        cs = jnp.where(b_zero, 1, jnp.where(a_lt_b, r * t, r))
        sn = jnp.where(b_zero, 0, jnp.where(a_lt_b, r, r * t))
        return cs, sn

    def _rotate_vector(self, vec, i, cs, sn):
        # Applies the Givens rotation `(cs, sn)` to the pair of entries `(i, i + 1)`
        # of `vec`.
        x1 = vec[i]
        y1 = vec[i + 1]
        x2 = cs.conj() * x1 - sn.conj() * y1
        y2 = sn * x1 + cs * y1
        return vec.at[i].set(x2).at[i + 1].set(y2)

    def transpose(self, state: _GMRESState, options: dict[str, Any]):
        transpose_options = {}
        if "preconditioner" in options:
            transpose_options["preconditioner"] = options["preconditioner"].transpose()
        operator = state
        return operator.transpose(), transpose_options

    def conj(self, state: _GMRESState, options: dict[str, Any]):
        conj_options = {}
        if "preconditioner" in options:
            conj_options["preconditioner"] = conj(options["preconditioner"])
        operator = state
        return conj(operator), conj_options

    def assume_full_rank(self):
        return True


GMRES.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
- `restart`: Size of the Krylov subspace built between restarts. The returned solution
    is the projection of the true solution onto this subpsace, so this direclty
    bounds the accuracy of the algorithm. Default is 20.
- `stagnation_iters`: The maximum number of iterations for which the solver may not
    decrease. If more than `stagnation_iters` restarts are performed without
    sufficient decrease in the residual, the algorithm is halted.
"""
