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


import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import lineax as lx
import numpy as np
import pytest

from .helpers import (
    construct_matrix,
    construct_singular_matrix,
    finite_difference_jvp,
    make_function_operator,
    make_matrix_operator,
    make_trivial_pytree_operator,
    tol,
    tree_allclose,
)


_full_rank_solvers_tags = [
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Cholesky(), lx.negative_semidefinite_tag),
    (lx.Cholesky(), lx.semidefinite_tag),
    (lx.Diagonal(), (lx.diagonal_tag, lx.hermitian_tag)),
    (lx.Circulant(rcond=1e-10), (lx.circulant_tag, lx.hermitian_tag)),
    (lx.HEVD(), lx.positive_semidefinite_tag),
    (lx.HEVD(), lx.negative_semidefinite_tag),
    (lx.HEVD(), lx.hermitian_tag),
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag),
    (lx.LU(), lx.hermitian_tag),
    (lx.QR(), lx.hermitian_tag),
    (lx.SVD(), lx.hermitian_tag),
    (lx.AutoLinearSolver(well_posed=True), lx.hermitian_tag),
    (lx.AutoLinearSolver(well_posed=False), lx.hermitian_tag),
]

# Solvers that return the pseudoinverse solution for a singular operator.
_singular_solvers_tags = [
    (lx.Diagonal(), (lx.diagonal_tag, lx.hermitian_tag)),
    (lx.Circulant(rcond=1e-10), (lx.circulant_tag, lx.hermitian_tag)),
    (lx.HEVD(), lx.positive_semidefinite_tag),
    (lx.HEVD(), lx.hermitian_tag),
    (lx.SVD(), lx.hermitian_tag),
    (lx.AutoLinearSolver(well_posed=False), lx.hermitian_tag),
]

_make_operators = [
    make_matrix_operator,
    make_trivial_pytree_operator,
    make_function_operator,
]


def _construct_singular(getkey, tags, num, dtype):
    # Any of `HEVD`, `Diagonal` or `Circulant` makes `construct_singular_matrix` zero a
    # mode of a square matrix, rather than trim it to a non-square (and so
    # non-Hermitian) one. Its tangents keep the rank constant; symmetrising them keeps
    # them Hermitian (as tangents inherit their primal's tags) and rank-preserving, as
    # `T + Tᴴ` vanishes on the null space wherever `T` does.
    matrix, *tangents = construct_singular_matrix(
        getkey, lx.HEVD(), tags, num=num, dtype=dtype
    )
    return matrix, *((t + t.conj().T) / 2 for t in tangents)


def _reference(matrix, vector):
    return jnp.vdot(vector, jnp.linalg.pinv(matrix) @ vector).real


def _form(make_operator, getkey, solver, tags):
    def fn(matrix, vector):
        operator = make_operator(getkey, matrix, tags)
        return lx.dual_hermitian_form(vector, operator, solver)

    return fn


@pytest.mark.parametrize("make_operator", _make_operators)
@pytest.mark.parametrize("solver, tags", _full_rank_solvers_tags)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_value(getkey, make_operator, solver, tags, dtype):
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=dtype)
    vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    out = _form(make_operator, getkey, solver, tags)(matrix, vector)
    assert out.dtype == jnp.float64
    assert tree_allclose(out, _reference(matrix, vector))


@pytest.mark.parametrize("make_operator", _make_operators)
@pytest.mark.parametrize("solver, tags", _singular_solvers_tags)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_value_singular(getkey, make_operator, solver, tags, dtype):
    (matrix,) = _construct_singular(getkey, tags, 1, dtype)
    vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    # The vector has a component outside the range of the operator.
    residual = vector - matrix @ jnp.linalg.pinv(matrix) @ vector
    assert jnp.linalg.norm(residual) > 1e-3
    out = _form(make_operator, getkey, solver, tags)(matrix, vector)
    assert tree_allclose(out, _reference(matrix, vector))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_unit_diagonal(dtype):
    vector = jnp.arange(1, 4).astype(dtype)
    operator = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), dtype))
    for solver in (lx.Diagonal(), lx.AutoLinearSolver(well_posed=True)):
        out = lx.dual_hermitian_form(vector, operator, solver)
        assert tree_allclose(out, jnp.array(14.0))


def test_pytree_vector(getkey):
    (a, b) = construct_matrix(getkey, lx.Cholesky(), lx.positive_semidefinite_tag, 2)
    matrix = {
        "a": {"a": a, "b": jnp.zeros((3, 2))},
        "b": {"a": jnp.zeros((2, 3)), "b": b[:2, :2]},
    }
    struct = {
        "a": jax.ShapeDtypeStruct((3,), jnp.float64),
        "b": jax.ShapeDtypeStruct((2,), jnp.float64),
    }
    operator = lx.PyTreeLinearOperator(matrix, struct, lx.positive_semidefinite_tag)
    vector = {"a": jr.normal(getkey(), (3,)), "b": jr.normal(getkey(), (2,))}
    out = lx.dual_hermitian_form(vector, operator)
    flat = jnp.concatenate([vector["a"], vector["b"]])
    assert tree_allclose(out, _reference(operator.as_matrix(), flat))


def test_state(getkey):
    solver = lx.Cholesky()
    (matrix,) = construct_matrix(getkey, solver, lx.positive_semidefinite_tag)
    operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    state = solver.init(operator, {})
    vector = jr.normal(getkey(), (3,))
    out = lx.dual_hermitian_form(vector, operator, solver, state=state)
    assert tree_allclose(out, _reference(matrix, vector))


def test_partial_isometry(getkey):
    # An orthogonal projector: `A⁺ = A`, so the form is `||Ax||²`.
    q, _ = jnp.linalg.qr(jr.normal(getkey(), (4, 2)))
    projector = q @ q.T
    tags = (lx.positive_semidefinite_tag, lx.partial_isometry_tag)
    vector = jr.normal(getkey(), (4,))

    def fn(vector):
        operator = lx.MatrixLinearOperator(projector, tags)
        return lx.dual_hermitian_form(vector, operator, lx.HEVD())

    assert tree_allclose(fn(vector), jnp.sum((projector @ vector) ** 2))
    assert tree_allclose(jax.grad(fn)(vector), 2 * projector @ vector)


def _check_jvp(fn, primals, tangents):
    out, t_out = jax.jvp(fn, primals, tangents)
    expected_out, expected_t_out = finite_difference_jvp(fn, primals, tangents)
    assert tree_allclose(out, expected_out)
    assert tree_allclose(t_out, expected_t_out, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("wrt", ("vector", "operator", "both"))
@pytest.mark.parametrize("make_operator", _make_operators)
@pytest.mark.parametrize("solver, tags", _full_rank_solvers_tags)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp(getkey, wrt, make_operator, solver, tags, dtype):
    matrix, t_matrix = construct_matrix(getkey, solver, tags, num=2, dtype=dtype)
    vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    t_vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    if wrt == "vector":
        t_matrix = jnp.zeros_like(t_matrix)
    elif wrt == "operator":
        t_vector = jnp.zeros_like(t_vector)
    fn = _form(make_operator, getkey, solver, tags)
    _check_jvp(fn, (matrix, vector), (t_matrix, t_vector))


@pytest.mark.parametrize("wrt", ("vector", "operator", "both"))
@pytest.mark.parametrize("make_operator", _make_operators)
@pytest.mark.parametrize("solver, tags", _singular_solvers_tags)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp_singular(getkey, wrt, make_operator, solver, tags, dtype):
    matrix, t_matrix = _construct_singular(getkey, tags, 2, dtype)
    vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    t_vector = jr.normal(getkey(), (matrix.shape[0],), dtype=dtype)
    if wrt == "vector":
        t_matrix = jnp.zeros_like(t_matrix)
    elif wrt == "operator":
        t_vector = jnp.zeros_like(t_vector)
    fn = _form(make_operator, getkey, solver, tags)
    _check_jvp(fn, (matrix, vector), (t_matrix, t_vector))


def test_jvp_operator_only(getkey):
    # Differentiating only the operator leaves the vector's tangent symbolically zero.
    solver = lx.HEVD()
    matrix, t_matrix = _construct_singular(getkey, lx.hermitian_tag, 2, jnp.float64)
    vector = jr.normal(getkey(), (matrix.shape[0],))
    fn = lambda m: _form(make_matrix_operator, getkey, solver, lx.hermitian_tag)(
        m, vector
    )
    _check_jvp(fn, (matrix,), (t_matrix,))


def _singular_family(getkey, dtype, size=4):
    """A Hermitian `A(θ) = V diag(d(θ)) Vᴴ` whose zero eigenvalue stays zero for all θ,
    so that its rank is constant to every order, and a vector `x(θ)` with a component
    in the null space of `A(θ)`."""
    v, _ = jnp.linalg.qr(jr.normal(getkey(), (size, size), dtype=dtype))
    d0 = jnp.array([2.0, -1.5, 0.7, 0.0])
    d1 = jnp.array([0.3, 0.5, -0.2, 0.0])
    x0 = jr.normal(getkey(), (size,), dtype=dtype)
    x1 = jr.normal(getkey(), (size,), dtype=dtype)

    def build(theta):
        d = (d0 + theta * d1 + 0.1 * theta**2 * d1).astype(dtype)
        matrix = (v * d[None, :]) @ v.conj().T
        theta = theta.astype(dtype)
        vector = x0 + theta * x1 + theta**2 * x0
        return matrix, vector

    return build


def _full_rank_family(getkey, dtype, size=3):
    a, b = construct_matrix(
        getkey, lx.Cholesky(), lx.positive_semidefinite_tag, num=2, dtype=dtype
    )
    x0 = jr.normal(getkey(), (size,), dtype=dtype)
    x1 = jr.normal(getkey(), (size,), dtype=dtype)

    def build(theta):
        theta = theta.astype(dtype)
        matrix = a + (theta + theta**2) * b
        vector = x0 + theta * x1 + theta**2 * x0
        return matrix, vector

    return build


_family_cases = [
    (_full_rank_family, lx.Cholesky(), lx.positive_semidefinite_tag),
    (_full_rank_family, lx.HEVD(), lx.positive_semidefinite_tag),
    (_full_rank_family, lx.LU(), lx.hermitian_tag),
    (_full_rank_family, lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag),
    (_singular_family, lx.HEVD(), lx.hermitian_tag),
    (_singular_family, lx.SVD(), lx.hermitian_tag),
]


def _scalar_fn(build, solver, tags):
    def fn(theta):
        matrix, vector = build(theta)
        operator = lx.MatrixLinearOperator(matrix, tags)
        return lx.dual_hermitian_form(vector, operator, solver)

    return fn


def _finite_difference(fn, theta):
    eps = np.sqrt(np.finfo(np.float64).eps)
    return (fn(theta + eps) - fn(theta - eps)) / (2 * eps)


@pytest.mark.parametrize("family, solver, tags", _family_cases)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_grad(getkey, family, solver, tags, dtype):
    fn = _scalar_fn(family(getkey, dtype), solver, tags)
    theta = jnp.array(0.3)
    grad = jax.grad(fn)(theta)
    assert tree_allclose(grad, _finite_difference(fn, theta), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("family, solver, tags", _family_cases)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_second_order(getkey, family, solver, tags, dtype):
    fn = _scalar_fn(family(getkey, dtype), solver, tags)
    theta = jnp.array(0.3)
    one = jnp.array(1.0)
    jvp_fn = lambda t: jax.jvp(fn, (t,), (one,))[1]
    _, jvp_jvp = jax.jvp(jvp_fn, (theta,), (one,))
    hessian = jax.hessian(fn)(theta)
    expected = _finite_difference(jax.grad(fn), theta)
    assert tree_allclose(jvp_jvp, expected, rtol=1e-4, atol=1e-6)
    assert tree_allclose(hessian, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("solver, tags", _full_rank_solvers_tags)
def test_vmap(getkey, solver, tags):
    matrix, matrix2 = construct_matrix(getkey, solver, tags, num=2)
    vectors = jr.normal(getkey(), (4, matrix.shape[0]))
    matrices = jnp.stack([matrix, matrix2])
    fn = _form(make_matrix_operator, getkey, solver, tags)

    out = jax.vmap(fn, in_axes=(None, 0))(matrix, vectors)
    expected = jnp.stack([_reference(matrix, v) for v in vectors])
    assert tree_allclose(out, expected)

    out = jax.vmap(fn, in_axes=(0, None))(matrices, vectors[0])
    expected = jnp.stack([_reference(m, vectors[0]) for m in matrices])
    assert tree_allclose(out, expected)

    out = jax.vmap(jax.grad(fn), in_axes=(0, None))(matrices, vectors[0])
    expected = jnp.stack([jax.grad(fn)(m, vectors[0]) for m in matrices])
    assert tree_allclose(out, expected)


def test_jit(getkey):
    solver = lx.HEVD()
    (matrix,) = _construct_singular(getkey, lx.hermitian_tag, 1, jnp.float64)
    vector = jr.normal(getkey(), (matrix.shape[0],))
    fn = jax.jit(_form(make_matrix_operator, getkey, solver, lx.hermitian_tag))
    assert tree_allclose(fn(matrix, vector), _reference(matrix, vector))


def test_non_hermitian_raises(getkey):
    operator = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    with pytest.raises(ValueError, match="Hermitian"):
        lx.dual_hermitian_form(jnp.ones(3), operator, lx.LU())


def test_raw_array_raises():
    # Under the test suite's runtime typechecking this is caught by the annotation.
    with pytest.raises((ValueError, jaxtyping.TypeCheckError)):
        lx.dual_hermitian_form(jnp.ones(3), jnp.eye(3))  # pyright: ignore


def test_mismatched_structure_raises(getkey):
    operator = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)
    with pytest.raises(ValueError, match="structure"):
        lx.dual_hermitian_form(jnp.ones(4), operator, lx.Cholesky())


def test_rank_compat_raises():
    operator = lx.MatrixLinearOperator(
        jnp.diag(jnp.array([1.0, 1.0, 0.0])), (lx.hermitian_tag, lx.MaxRankTag(2))
    )
    with pytest.raises(ValueError, match="full rank"):
        lx.dual_hermitian_form(jnp.ones(3), operator, lx.LU())


def _count_solves(jaxpr):
    count = 0
    for eqn in jaxpr.eqns:
        count += eqn.primitive.name == "linear_solve"
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else (param,):
                if isinstance(sub, jax_core.ClosedJaxpr):
                    count += _count_solves(sub.jaxpr)
                elif isinstance(sub, jax_core.Jaxpr):
                    count += _count_solves(sub)
    return count


@pytest.mark.parametrize("solver", (lx.HEVD(), lx.SVD()))
def test_full_rank_tag_jvp(getkey, solver):
    # Declared full rank, a rank-deficient-capable solver skips the correction: the
    # result is unchanged, with one solve fewer.
    matrix, t_matrix = construct_matrix(getkey, solver, lx.hermitian_tag, num=2)
    vector = jr.normal(getkey(), (matrix.shape[0],))

    def jvp(tags):
        def fn(m):
            operator = lx.MatrixLinearOperator(m, (lx.hermitian_tag, *tags))
            return lx.dual_hermitian_form(vector, operator, solver)

        return lambda m, t: jax.jvp(fn, (m,), (t,))

    plain, tagged = jvp(()), jvp((lx.RankTag(matrix.shape[0]),))
    assert tree_allclose(plain(matrix, t_matrix), tagged(matrix, t_matrix))
    num_plain = _count_solves(jax.make_jaxpr(plain)(matrix, t_matrix).jaxpr)
    num_tagged = _count_solves(jax.make_jaxpr(tagged)(matrix, t_matrix).jaxpr)
    assert num_tagged == num_plain - 1
