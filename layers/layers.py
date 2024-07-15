import torch
import geoopt
import torch.nn.functional as F
from geoopt.manifolds.stereographic.math import _project
import math

def fix_nan(res, eps = 1e-15):
    res = torch.nan_to_num(res, nan=eps)
    res_sign = torch.sign(res)
    res = res_sign * torch.abs(res).clamp_min(eps)
    return res

def fix_zero(res, eps = 1e-15):
    res_sign = torch.sign(res)
    return res_sign * torch.abs(res).clamp_min(eps)

# package.nn.modules.py
def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.
    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.
    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float
    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)
    # else trust input
    return ball

class ClipNorm(torch.nn.Module):
    def __init__(self, max_norm=0.8, dimensions_per_space=None):
        super().__init__()
        self.max_norm = max_norm
        self.dimension_per_space = dimensions_per_space

    def get_mean_norm(self, input):
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def forward(self, input):  # input bs x in_feat
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        input_l2 = torch.norm(rs_input, p=2, dim=-1, keepdim=True)
        clipped_input = torch.minimum(self.max_norm / input_l2,
                                      torch.ones_like(input_l2)) * rs_input
        if self.dimension_per_space:
            clipped_input = clipped_input.view(*input_shape)
        return clipped_input

# package.nn.functional.py
def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        # hype_bias = ball.expmap0(bias)
        hype_bias = bias
        output = ball.mobius_add(output, hype_bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output
    
class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, dropout=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = create_ball(ball, c)
        self.dropout = dropout
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        return mobius_linear(
            input,
            weight=drop_weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


# # Lorentz to Poincare
# def mobius_linear_l2p(input, weight, scale, ball, dropout, training, bias, nonlin=None,):
#     K = ball.k
#     sqrt_K = torch.abs(K).sqrt()
#     eps = 1e-8

#     x_norm_2 = (input * input).sum(dim=-1, keepdim=True)
#     x_hat_numerator = (1 - K * x_norm_2) / fix_zero(sqrt_K + K * sqrt_K * x_norm_2, eps)
#     x_hat_denom = (2 * input) / fix_zero(1 + K * x_norm_2, eps)
#     x_hat = torch.cat([x_hat_numerator, x_hat_denom], dim=-1)

#     wx = F.linear(x_hat, weight, bias)
#     if dropout is not None:
#         wx = F.dropout(wx, dropout, training=training)
#     if nonlin is not None:
#         wx = nonlin(wx)
    
#     x_t = wx.narrow(-1, 0, 1).sigmoid() * scale.exp() + (1/sqrt_K) + eps
#     wx_s = wx.narrow(-1, 1, wx.shape[-1] - 1)
#     frac = ((x_t * x_t) + 1/K).clamp_min(-1/K + eps) / fix_zero((wx_s * wx_s).sum(dim=-1, keepdim=True), eps)
#     x_s = wx_s * frac.sqrt()

#     output = x_s / fix_zero(1 + sqrt_K * x_t, eps)
#     return output
    
# class MobiusLinearL2P(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, nonlin=None, ball=None, c=1.0, dropout=0.) -> None:
#         super(MobiusLinearL2P, self).__init__()
#         self.ball = create_ball(ball, c)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.nonlin = nonlin
#         self.weight = torch.nn.Parameter(torch.empty((out_features + 1, in_features + 1)))
#         if bias:
#             self.bias = torch.nn.Parameter(torch.empty(out_features + 1))
#         else:
#             self.register_parameter('bias', None)
#         scale = 1000
#         self.scale = torch.nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=True)
#         self.reset_parameters()

#     @torch.no_grad()
#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             torch.nn.init.uniform_(self.bias, -bound, bound)
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return mobius_linear_l2p(
#             input,
#             weight=self.weight,
#             ball=self.ball,
#             scale=self.scale,
#             dropout=self.dropout,
#             training=self.training,
#             bias=self.bias,
#             nonlin=self.nonlin,
#         )

#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )


# # Lorentz to Poincare
# def mobius_linear_l2p(input, ball, w, v, bias_w, bias_v, dropout, training, nonlin):
#     K = ball.k
#     sqrt_K = torch.abs(K).sqrt()
#     eps = 1e-15
#     x_norm_2 = (input * input).sum(dim=-1, keepdim=True)
#     x_hat_numerator = (1. - K * x_norm_2) / fix_zero(sqrt_K + K * sqrt_K * x_norm_2, eps)
#     x_hat_denom = (2. * input) / fix_zero(1. + K * x_norm_2, eps)
#     x_hat = torch.cat([x_hat_numerator, x_hat_denom], dim=-1)

#     vx = torch.sigmoid((v @ x_hat.T).T + bias_v) + 1/sqrt_K
#     vx_norm_2 = vx.pow(2).sum(dim=-1, keepdim=True).clamp_min(-1/K + 1)
#     vx_norm = (vx_norm_2 + 1/K).sqrt()

#     wx = (w @ x_hat.T).T + bias_w
#     wx_norm_2 = wx.pow(2).sum(dim=-1, keepdim=True)
#     wx_norm = (wx_norm_2).sqrt()

#     frac = vx_norm / fix_zero(wx_norm, eps)
#     mx = wx * frac

#     if dropout is not None:
#         mx = F.dropout(mx, dropout, training=training)
#     if nonlin is not None:
#         mx = nonlin(mx)

#     mx_norm_2 = mx.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
#     output = mx / fix_zero(1 + (mx_norm_2 - 1/K).sqrt()*sqrt_K, eps)
#     return output
    
# class MobiusLinearL2P(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, nonlin=None, ball=None, c=1.0, dropout=0.) -> None:
#         super(MobiusLinearL2P, self).__init__()
#         self.ball = create_ball(ball, c)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.nonlin = nonlin
#         self.w = torch.nn.Parameter(torch.empty((out_features, in_features + 1)))
#         self.v = torch.nn.Parameter(torch.empty((1, in_features + 1)))
#         if bias:
#             self.bias_w = torch.nn.Parameter(torch.empty(out_features))
#             self.bias_v = torch.nn.Parameter(torch.empty(1))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     @torch.no_grad()
#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
#         if self.bias_w is not None and self.bias_v is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             torch.nn.init.uniform_(self.bias_w, -bound, bound)
#             torch.nn.init.uniform_(self.bias_v, -bound, bound)
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         support = mobius_linear_l2p(
#             input,
#             ball=self.ball,
#             w=self.w,
#             v=self.v,
#             bias_w=self.bias_w,
#             bias_v=self.bias_v,
#             dropout=self.dropout,
#             training=self.training,
#             nonlin=self.nonlin
#         )
#         return support

#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )

# Lorentz to Poincare
def mobius_linear_fast(input, weight, ball, dropout, training, bias, nonlin=None,):
    K = ball.k
    sqrt_K = torch.abs(K).sqrt()
    eps = 1e-15

    x_norm_2 = (input * input).sum(dim=-1, keepdim=True)
    x_hat_numerator = (1 - K * x_norm_2) / fix_zero(sqrt_K + K * sqrt_K * x_norm_2, eps)
    x_hat_denom = (2 * input) / fix_zero(1 + K * x_norm_2, eps)
    x_hat = torch.cat([x_hat_numerator, x_hat_denom], dim=-1)

    wx = F.linear(x_hat, weight, bias)
    if dropout is not None:
        wx = F.dropout(wx, dropout, training=training)
    if nonlin is not None:
        wx = nonlin(wx)

    wx_norm_2 = (wx * wx).sum(dim=-1, keepdim=True)
    output = wx / fix_zero(1 + (wx_norm_2 - 1/K).sqrt()*sqrt_K , eps)
    return _project(output, K, dim=-1)
    
class MobiusLinearFast(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, nonlin=None, ball=None, c=1.0, dropout=0.) -> None:
        super(MobiusLinearFast, self).__init__()
        if ball is not None:
            self.ball = ball
        else:
            self.ball = create_ball(ball, c)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.nonlin = nonlin
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features + 1)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return mobius_linear_fast(
            input,
            weight=self.weight,
            ball=self.ball,
            dropout=self.dropout,
            training=self.training,
            bias=self.bias,
            nonlin=self.nonlin,
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# HNN ++ layer
def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r
    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    res = 2. * z_norm / rc * arsinh(
        (2. * torch.matmul(rcx, z_unit) * drcr.cosh() - (1. + cx2) * drcr.sinh())
        / torch.clamp_min(1. - cx2, 1e-15))
    # fix instability
    res = torch.nan_to_num(res, nan=1e-14)
    res_sign = torch.sign(res)
    res = res_sign * torch.abs(res).clamp_min(1e-14)
    return res

def poincare_linear(support, weight_g, weight_v, bias, c, out_split : int = 1, dropout = None, training = None):
    rc = c.sqrt()
    support = unidirectional_poincare_mlr(support, weight_g, weight_v, bias, c)
    support = (rc * support).sinh() / rc
    if out_split > 1:
        size = support.size()
        support = support.view(*size[:-1], out_split, size[-1] // out_split)
    if dropout is not None:
        support = F.dropout(support, dropout, training=training)
    support = support / (1 + (1 + c * support.pow(2).sum(dim=-1, keepdim=True)).sqrt())
    return _project(support, -c, dim=-1)

class MobiusLinearPlus(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, dropout=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.dropout = dropout
        self.ball = create_ball(ball, c)

        out_split = 1.; gain = 1.
        std = (2 * self.in_features * self.out_features / out_split) ** -0.5 * gain
        weight = torch.empty(self.in_features, self.out_features).normal_(mean=0, std=std)
        self.weight_g = torch.nn.Parameter(weight.norm(dim=0)) # in_features x out_features
        self.weight_v = torch.nn.Parameter(weight) # in_features x out_features

        bias = torch.nn.Parameter(torch.empty(self.out_features), requires_grad=True)
        self.bias = bias
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        w_g = self.weight_g
        w_v = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)
        output = poincare_linear(
            input,
            weight_g=w_g,
            weight_v=w_v,
            bias=self.bias,
            c=self.ball.c,
            dropout=self.dropout,
            training=self.training,
        )
        return output

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()