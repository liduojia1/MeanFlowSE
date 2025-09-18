import abc
import torch
import numpy as np

from flowmse.util.registry import Registry


ODEsolverRegistry = Registry("ODEsolver")


class ODEsolver(abc.ABC):
    def __init__(self, ode, VF_fn):
        super().__init__()
        self.ode = ode
        self.VF_fn = VF_fn

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        pass


@ODEsolverRegistry.register('euler')
class EulerODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t, y, stepsize, *args):
        dt = -stepsize
        vectorfield = self.VF_fn(x, t, y)   # r=None -> 等价 r=t
        x = x + vectorfield * dt
        return x


# -------------------- 新增：MeanFlow Euler（位移式） --------------------
@ODEsolverRegistry.register('euler_mf')
class EulerMFODESolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t, y, stepsize, *args):
        Delta = stepsize
        r = (t - Delta).clamp_min(0.0)
        u = self.VF_fn(x, t, y, r)  # 传入 r
        return x - Delta * u
# ----------------------------------------------------------------------
