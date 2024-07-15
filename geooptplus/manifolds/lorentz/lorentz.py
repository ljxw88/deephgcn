import torch.nn
from typing import Tuple, Optional
from . import lmath as math
import geoopt
from geoopt import Manifold
from geoopt.utils import size2shape
from .utils import acosh


def arcosh(x: torch.Tensor):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z).to(dtype)


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k, learnable)
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        self.max_norm = 1000
        self.min_norm = 1e-8

    def get_curvature(self):
        return self.k

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        dn = x.size(dim) - 1
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(
            dim=dim, keepdim=True
        )
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        if not ok:
            reason = f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        inner_ = math.inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        if not ok:
            reason = "Minkowski inner produt is not equal to zero"
        else:
            reason = None
        return ok, reason

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x = x.clone()
        # x.narrow(-1, 0, 1).mul_(-1)
        # return torch.sqrt(self.k) * acosh(-(x.matmul(y.transpose(-1, -2))) / self.k)
        return math.cdist(x, y, k=self.k)
        # return -(x.matmul(y.transpose(-1, -2))) / self.k

    def sqdist(self, x: torch.Tensor, y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return -2 - 2 * math.inner(x, y)

    def lorentz_to_klein(self, x): # same as h2h-gcn
        dim = x.shape[-1] - 1
        return acosh(x.narrow(-1, 1, dim) / x.narrow(-1, 0, 1))

    def klein_to_lorentz(self, x): # basically same except for numerical instability settings
        norm = (x * x).sum(dim=-1, keepdim=True)
        size = x.shape[:-1] + (1, )
        return torch.cat([x.new_ones(size), x], dim=-1) / torch.clamp_min(torch.sqrt(1 - norm), 1e-7)

    def lorentz_to_poincare(self, x):
        return math.lorentz_to_poincare(x, self.k)

    def poincare_to_lorentz(self, x):
        return math.poincare_to_lorentz(x, self.k)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def proju0(self, v: torch.Tensor) -> torch.Tensor:
        v = math.project_u0(v)
        return v

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1
    ) -> torch.Tensor:
        if norm_tan is True:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project is True:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    def clogmap(self, x, y):
        return math.clogmap(x, y)

    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        # TODO: x argument for maintaining the support of optims
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(self, v: torch.Tensor = None, *, keepdim=False, dim=-1,) -> torch.Tensor:
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor):
        # x = x.clone()
        # x.narrow(-1, 0, 1).mul_(-1)
        # return x @ y.transpose(-1, -2)
        return math.cinner(x, y)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def mobius_add(self, x, y):
        v = self.logmap0(y)
        v = self.transp0(x, v)
        return self.expmap(x, v)
    
    def mobius_scalar_mul(self, r, x):
        v = self.logmap0(x)
        support = r * v
        return self.expmap0(support)

    def normalize_tangent(self, p, p_tan, c):
        """
        Normalize tangent vectors to place the vectors satisfies <p, p_tan>_L=0
        :param p: the tangent spaces at p. size:[nodes, feature]
        :param p_tan: the tangent vector in tangent space at p
        """
        d = p_tan.size(1) - 1
        p_tail = p.narrow(1, 1, d)
        p_tan_tail = p_tan.narrow(1, 1, d)
        ptpt = torch.sum(p_tail * p_tan_tail, dim=1, keepdim=True)
        p_head = torch.sqrt(c + torch.sum(torch.pow(p_tail, 2), dim=1, keepdim=True) + self.eps[p.dtype])
        return torch.cat((ptpt / p_head, p_tan_tail), dim=1)

    def l_inner(self, x, y, keep_dim=False):
        # input shape [node, features]
        d = x.size(-1) - 1
        xy = x * y
        xy = torch.cat((-xy.narrow(1, 0, 1), xy.narrow(1, 1, d)), dim=1)
        return torch.sum(xy, dim=1, keepdim=keep_dim)

    def induced_distance(self, x, y, c):
        xy_inner = self.l_inner(x, y)
        sqrt_c = c ** 0.5
        return sqrt_c * arcosh(-xy_inner / c + self.eps[x.dtype])

    def normalize_tangent_zero(self, p_tan, c):
        zeros = torch.zeros_like(p_tan)
        zeros[:, 0] = c ** 0.5
        return self.normalize_tangent(zeros, p_tan, c)

    def normalize(self, p, c):
        """
        Normalize vector to confirm it is located on the hyperboloid
        :param p: [nodes, features(d + 1)]
        :param c: parameter of curvature
        """
        d = p.size(-1) - 1
        narrowed = p.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = c + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        first = torch.sqrt(first)
        return torch.cat((first, narrowed), dim=1)

    def log_map_x(self, x, y, c, is_tan_normalize=True):
        """
        Logarithmic map at x: project hyperboloid vectors to a tangent space at x
        :param x: vector on hyperboloid
        :param y: vector to project a tangent space at x
        :param normalize: whether normalize the y_tangent
        :return: y_tangent
        """
        xy_distance = self.induced_distance(x, y, c)
        tmp_vector = y + self.l_inner(x, y, keep_dim=True) / c * x
        tmp_norm = torch.sqrt(self.l_inner(tmp_vector, tmp_vector) + self.eps[x.dtype])
        y_tan = xy_distance.unsqueeze(-1) / tmp_norm.unsqueeze(-1) * tmp_vector
        if is_tan_normalize:
            y_tan = self.normalize_tangent(x, y_tan, c)
        return y_tan

    def log_map_zero(self, y, c, is_tan_normalize=True):
        zeros = torch.zeros_like(y)
        zeros[:, 0] = c ** 0.5
        return self.log_map_x(zeros, y, c, is_tan_normalize)

    def exp_map_x(self, p, dp, c, is_res_normalize=True, is_dp_normalize=True):
        if is_dp_normalize:
            dp = self.normalize_tangent(p, dp, c)
        dp_lnorm = self.l_inner(dp, dp, keep_dim=True)
        dp_lnorm = torch.sqrt(torch.clamp(dp_lnorm + self.eps[p.dtype], 1e-6))
        dp_lnorm_cut = torch.clamp(dp_lnorm, max=50)
        sqrt_c = c ** 0.5
        res = (torch.cosh(dp_lnorm_cut / sqrt_c) * p) + sqrt_c * (torch.sinh(dp_lnorm_cut / sqrt_c) * dp / dp_lnorm)
        if is_res_normalize:
            res = self.normalize(res, c)
        return res

    def exp_map_zero(self, dp, c, is_res_normalize=True, is_dp_normalize=True):
        zeros = torch.zeros_like(dp)
        zeros[:, 0] = c ** 0.5
        return self.exp_map_x(zeros, dp, c, is_res_normalize, is_dp_normalize)

    def matvec_regular(self, m, x, b, c, use_bias):
        d = x.size(1) - 1
        x_tan = self.log_map_zero(x, c)
        x_head = x_tan.narrow(1, 0, 1)
        x_tail = x_tan.narrow(1, 1, d)
        mx = x_tail @ m.transpose(-1, -2)
        if use_bias:
            mx_b = mx + b
        else:
            mx_b = mx
        mx = torch.cat((x_head, mx_b), dim=1)
        mx = self.normalize_tangent_zero(mx, c)
        mx = self.exp_map_zero(mx, c)
        cond = (mx==0).prod(-1, keepdim=True, dtype=torch.uint8)
        res = torch.zeros(1, dtype=mx.dtype, device=mx.device)
        res = torch.where(cond, res, mx)
        return res

    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            zero point on the manifold
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    def weighted_midpoint(self, x, weights=None, reducedim = None):
        if weights is not None:
            ave = weights.matmul(x).sum(reducedim, keepdim = True)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, keepdim=True)).sum(reducedim, keepdim = True)
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return torch.sqrt(torch.abs(self.get_curvature())) * ave / denom

    retr = expmap
