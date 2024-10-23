from abc import ABCMeta, abstractmethod
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
import math
from sklearn.neighbors import NearestNeighbors
from utils import *
from torch import Tensor
from typing import List, Tuple

from Models import *


class CFE(metaclass=ABCMeta):
    '''
    Abstract class for Counterfactual Explanation.
    '''
    def __init__(self, model: nn.Module, mins: Tensor, maxs: Tensor) -> None:
        '''
        Initialize the CFE class.

        Arguments:
            model:      nn.Module, the model to be explained
            mins:       Tensor of the same shape as the input data, minimum
                        values for each feature used for scaling the data to [0, 1]
            maxs:       Tensor of the same shape as the input data, maximum
                        values for each feature used for scaling the data to [0, 1]
        '''
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        self.mins = mins.to(self.device, self.dtype)[None, ...]
        self.maxs = maxs.to(self.device, self.dtype)[None, ...]


    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        '''
        Move all attributes that are tensors and the model to the specified device and dtype.

        Arguments:
            device:     torch.device | str | None, device to move the attributes to
            dtype:      torch.dtype | None, dtype to move the attributes to
        '''
        assert device is None or isinstance(device, torch.device) or isinstance(device, str),\
            "device must be a torch.device, a string or None"
        assert dtype is None or isinstance(dtype, torch.dtype),\
            "dtype must be a torch.dtype or None"
        self.device = device
        self.dtype = dtype
        self.model.to(device, dtype)
        for key in self.__dict__.keys():
            if torch.is_tensor(self.__dict__[key]):
                self.__dict__[key] = self.__dict__[key].to(device, dtype)
    

    def predict(self, x: Tensor) -> Tensor:
        '''
        Renormalize the data to the original scale and evaluate the model.

        Arguments:
            x: Tensor of shape (n, d), the data to be evaluated
        Returns:
            Tensor of shape (n, 1), the predictions of the model
        '''
        try:
            return self.model(x * (self.maxs - self.mins) + self.mins)
        except Exception as e:
            print('The model is assumed to expect a tensor of shape \
                   (batch size, num features) as input. The following error \
                   may be caused by the model not expecting flat tensors.')
            raise e


    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Compute the counterfactuals.

        Arguments:
            x: Tensor of shape (n, d), the original data
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, d), the counterfactuals
        '''
        assert x.shape[1:] == self.mins.shape[1:], "Data is expected to have the same shape as the mins and maxs"
        # work in [0, 1] and flatten
        x = ((x.clone().to(self.device) - self.mins) / (self.maxs - self.mins))
        y = y.clone().to(self.device)
        cfs = self.get_CFs(x, y)
        # rescale and unflatten
        return cfs * (self.maxs - self.mins) + self.mins
    
    
    @abstractmethod
    def get_CFs(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Abstract method for computing the counterfactuals.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor, counterfactuals
        '''
        pass



class APG0_CFE(CFE):
    '''
    Class for computing counterfactual explanations using the APG0 algorithm.
    '''
    def __init__(self, model: nn.Module, mins: Tensor, maxs: Tensor, numclasses: int,
                 range_min: Tensor | None = None, range_max: Tensor | None = None,
                 L0: float = 1.0, lam0: float = 1.0, c: float = 0.01, beta: float = 0.5,
                 iters: int = 100, max_i: int = 100, eta: float = 1.1, prox: str = 'zero',
                 linesearch: bool = False, lam_steps: int = 10, verbose: bool = False,
                 scale_model: bool = True) -> None:
        '''
        Initialize the APG0_CFE class.

        Arguments:
            model:        nn.Module, the model to be explained
            mins:         Tensor, minimum values for each feature used for
                          scaling the data to [0, 1]
            maxs:         Tensor, maximum values for each feature used for
                          scaling the data to [0, 1]
            numclasses:   int, number of classes
            range_min:    Tensor | None, minimum values for the perturbations.
                          If None, the minimum values are set to 0.
            range_max:    Tensor | None, maximum values for the perturbations.
                          If None, the maximum values are set to 1.
            L0:           float, Lipschitz constant / step size, depending on
                          the boolean linesearch
            lam0:         float, regularization parameter. If 0, the model
                          is not called
            c:            float, constant term in the loss function
            beta:         float, parameter for the proximal operator
            iters:        int, number of iterations
            max_i:        int, maximum number of iterations for the line search
            eta:          float, eta for the line search
            prox:         str, proximal operator to use. Options: 'zero', 'half',
                          'one', 'clamp', 'zero_fixed', where 'zero_fixed' 
                          projects to a 0-norm ball of radius beta and 'clamp'
                          only clamps the values to the valid range.
            linesearch:   bool, if True, approximates the Lipschitz constant L
                          using backtracking line search in each iteration and
                          uses 1/L as the step size. if False, uses a quadratically
                          decaying step size with initial value L0.
            lam_steps:    int, number of steps for the section search
            verbose:      bool, whether to print the current section search step
                          and iteration
            scale_model:  bool, set to true to scale the model output from [0, 1] to
                          [-1, 1] for binary classification. (set to false if the model
                          output is already in [-1, 1])
        '''

        assert range_min is None or (range_min.min() >= -1 and range_min.max() < 1),\
            "Min ranges are expected to be in [0, 1)"
        assert range_max is None or (range_max.min() > -1 and range_max.max() <= 1),\
            "Max ranges are expected to be in (0, 1]"
        assert eta > 1, "Eta is expected to be greater than 1"
        assert prox in ['zero', 'half', 'one', 'clamp', 'zero_fixed'],\
            "prox is expected to be one of 'zero', 'half', 'one', 'clamp', 'zero_fixed',"\
            + f". Got {prox}."
        
        super().__init__(model, mins, maxs)
        self.numclasses = numclasses
        self.lam0 = lam0
        self.lam = lam0
        self.c = c
        self.beta = beta
        self.iters = iters
        if range_min is None:
            range_min = self.mins.clone().view(self.mins.size(0), -1)
        else:
            range_min = range_min.to(self.device, self.dtype)[None, ...]
        if range_max is None:
            range_max = self.maxs.clone().view(self.maxs.size(0), -1)
        else:
            range_max = range_max.to(self.device, self.dtype)[None, ...]
        self.range_min = range_min
        self.range_max = range_max
        self.L0 = L0
        self.max_i = max_i
        self.eta = eta
        self.linesearch = linesearch
        self.prox = {'zero': self.prox_0, 'twothirds': self.prox_twothirds, 
                     'half': self.prox_half, 'one': self.prox_one,
                     'zero_fixed': self.prox_0_fixed, 'clamp': self.clamp_only}[prox]
        self.maxL = 1e4
        self.lam_steps = lam_steps
        self.verbose = verbose
        self.scale_model = scale_model


    def predict(self, x: Tensor) -> Tensor:
        '''
        Renormalize the data to the original scale and evaluate the model.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
        Returns:
            Tensor of shape (n, 1)
        '''
        if self.numclasses == 2:
            if self.scale_model:
                return 2 * super().predict(x) - 1
            else:
                return super().predict(x)
        else:
            return super().predict(x)


    def loss(self, x: Tensor, w: Tensor, y: Tensor, active: Tensor | None = None) -> Tensor:
        '''
        Compute the l2 regularized classification loss.

        Arguments:
            x:      Tensor of shape (n, d) expected to be in [0, 1],
                    the original data
            w:      Tensor of shape (n, d), the perturbation
            y:      Tensor of shape (n, 1) expected to be in {-1, 1},
                    the target label
            active: Tensor | None, mask indicating active elements.
        Returns:
            Tensor of shape (n, 1)
        '''
        if active is None:
            active = torch.ones_like(self.lam, dtype=torch.bool)
        if self.numclasses == 2 and self.lam0 != 0:
            classloss = self.lam[active.view(-1)] * torch.relu(-y * self.predict(x + w) + self.c)
        elif self.lam0 != 0:
            logits = self.predict(x + w)
            one_hot_y = F.one_hot(y.view(-1), self.numclasses)
            Z_t = torch.sum(logits * one_hot_y, dim=1, keepdim=True)
            Z_i = torch.amax(logits * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1, keepdim=True)
            classloss = self.lam[active.view(-1)] * F.relu(Z_i - Z_t + self.c)
        else:
            classloss = 0
        return classloss + (w ** 2).view(w.size(0), -1).sum(dim=-1, keepdim=True)
    

    def prox_0(self, w: Tensor, x: Tensor, L: Tensor) -> Tensor:
        '''
        Compute the proximal operator of the 0-norm regularizer
        and the range constraints.

        Arguments:
            w: Tensor of shape (n, d), the perturbation
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            L: Tensor of shape (n, 1), the Lipschitz constant
        Returns:
            Tensor of shape (n, d)
        '''
        w_clamp = torch.minimum(torch.maximum(w, self.range_min), self.range_max)
        w_clamp = torch.clamp(x + w_clamp, 0, 1) - x
        mask = w ** 2 - (w_clamp - w) ** 2 < 2 * self.beta / L
        w_clamp[mask] = 0
        return w_clamp
    

    def prox_half(self, w: Tensor, x: Tensor, L: Tensor) -> Tensor:
        '''
        Compute the proximal operator of the 1/2-quasinorm regularizer
        and clip the values to the valid range.

        Arguments:
            w: Tensor of shape (n, d), the perturbation
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            L: Tensor of shape (n, 1), the Lipschitz constant
        Returns:
            Tensor of shape (n, d)
        '''
        lam = self.beta / L
        p_lam = (54 ** (1 / 3) / 4) * lam ** (2 / 3)

        mask1 = (w > p_lam)
        mask2 = (torch.abs(w) <= p_lam)
        mask3 = (w < -p_lam)
        mask4 = mask1 | mask3

        phi_lam_x = torch.arccos((lam / 8) * (torch.abs(w) / 3) ** (-1.5))

        w[mask4] = ((2 / 3) * torch.abs(w[mask4])
                    * (1 + torch.cos((2 * math.pi) / 3
                    - (2 * phi_lam_x[mask4]) / 3)))
        w[mask3] = -w[mask3]
        w[mask2] = 0

        w = torch.minimum(torch.maximum(w, self.range_min), self.range_max)
        return torch.clamp(x + w, 0, 1) - x
    

    def prox_one(self, w, x, L):
        '''
        Compute the proximal operator of the 1-norm regularization term
        and clip the values to the valid range.

        Arguments:
            w: Tensor of shape (n, d), the perturbation
            x: Tensor of shape (n, d), the original data
            L: Tensor of shape (n, 1), the Lipschitz constant
        Returns:
            Tensor of shape (n, d)
        '''
        w_hat = w.sign() * torch.relu(torch.abs(w) - self.beta / L)
        w_hat = torch.minimum(torch.maximum(w_hat, self.range_min), self.range_max)
        return torch.clamp(x + w_hat, 0, 1) - x


    def clamp_only(self, w, x, L):
        '''
        Compute the proximal operator of the 1-norm regularization term
        and clip the values to the valid range.

        Arguments:
            w: Tensor of shape (n, d), the perturbation
            x: Tensor of shape (n, d), the original data
            L: Tensor of shape (n, 1), the Lipschitz constant
        Returns:
            Tensor of shape (n, d)
        '''
        w = torch.minimum(torch.maximum(w, self.range_min), self.range_max)
        return torch.clamp(x + w, 0, 1) - x


    def prox_0_fixed(self, w: Tensor, x: Tensor, L: Tensor) -> Tensor:
        '''
        Compute the proximal operator of the 0-norm regularizer
        and the range constraints.

        Arguments:
            w: Tensor of shape (n, d), the perturbation
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            L: Tensor of shape (n, 1), the Lipschitz constant
        Returns:
            Tensor of shape (n, d)
        '''
        w_clamp = torch.minimum(torch.maximum(w, self.range_min), self.range_max)
        w_clamp = torch.clamp(x + w_clamp, 0, 1) - x
        _, idx = torch.topk(w ** 2 - (w_clamp - w) ** 2, k=int(self.beta), dim=-1)
        w_res = torch.zeros_like(w)
        w_res[torch.arange(x.size(0), device=self.device).view(-1, 1), idx] = w_clamp[torch.arange(x.size(0), device=self.device).view(-1, 1), idx]
        return w_res
    

    def get_grad_loss(self, x: Tensor, w: Tensor, y: Tensor) -> Tensor:
        '''
        Compute the gradient and loss.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            w: Tensor of shape (n, d), the perturbation
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tuple of Tensors of shape (n, d) and (n, 1)
        '''
        with torch.enable_grad():
            w.requires_grad = True
            loss = self.loss(x, w, y)
            grad = torch.autograd.grad(loss.sum(), w)[0]
        w.detach_()
        return grad, loss.detach()
    

    def backtrack_line_search(self, x: Tensor, w: Tensor, y: Tensor, grad: Tensor, loss: Tensor, L: Tensor) -> Tensor:
        '''
        Compute an approximation of the Lipschitz constant at x + w.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            w: Tensor of shape (n, d), the perturbation
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
            grad: Tensor of shape (n, d), the gradient
            loss: Tensor of shape (n, 1), the loss
            L: Tensor of shape (n, 1), the previous Lipschitz constants
        Returns:
            Tensor of shape (n, 1) containing the Lipschitz constants
        '''
        
        L_res = L.clone()
        active = torch.ones_like(y, dtype=torch.bool)

        def QF(L_: float) -> Tuple[Tensor, Tensor]:
            prx = self.prox(w - (1 / L_) * grad, x, L_)
            val = loss + ((prx - w) * grad).view(w.size(0), -1).sum(dim=-1, keepdim=True)
            val += L_ / 2 * ((prx - w) ** 2).view(w.size(0), -1).sum(dim=-1, keepdim=True)
            return val, self.loss(x, prx, y, active)
        
        for i in range(self.max_i):
            L_new = torch.clamp(self.eta ** i * L, max=self.maxL)
            Q, F = QF(L_new)
            mask = (Q >= F)[:, 0]

            if mask.any():
                L_res[active.nonzero()[..., 0][mask]] = L_new[mask]
                mask = ~mask
                x, w, y, grad, loss, L = x[mask], w[mask], y[mask], grad[mask], loss[mask], L[mask]
                active[active.clone()] = mask
            if not active.any():
                break

        return L_res

    
    def get_CFs(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Compute the counterfactuals using the APG0 algorithm.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, d)
        '''
        assert x.min() >= 0 and x.max() <= 1, "Data is expected to be in [0, 1]"
        assert x.shape[0] == y.shape[0], "Data and target label are expected to have the same number of samples"
        assert len(y.shape) == 2 and y.shape[1] == 1, "Target label is expected to be a tensor of shape (n, 1)"
        if self.numclasses == 2:
            mask1 = y == 1
            mask2 = y == -1
            assert (mask1 | mask2).all(), "Classes are expected to be in {-1, 1}"
        else:
            mask = y.view(y.size(0), 1) == torch.arange(self.numclasses, device=self.device, dtype=y.dtype).view(1, -1)
            assert (mask.any(-1)).all(), "Classes are expected to be in {0, ..., numclasses-1}"

        best_w = torch.zeros_like(x)
        best_norm = torch.full((x.size(0),), torch.inf, dtype=self.dtype, device=self.device)
        self.lam = torch.full((x.size(0), 1), self.lam0, device=self.device, dtype=self.dtype)
        lam_lb = torch.zeros_like(self.lam)
        lam_ub = torch.full_like(self.lam, 1e10)

        for step in range(self.lam_steps):
            w = torch.zeros_like(x)
            v = torch.zeros_like(x)
            v_old = torch.zeros_like(x)
            t = 1.0
            t_old = 1.0
            L0 = self.L0 if self.linesearch else 1 / self.L0
            L = torch.full_like(y, L0, dtype=self.dtype).view(y.size(0), *([1] * (x[0].dim())))

            for i in range(self.iters):
                if self.verbose:
                    s1 = ''.join([' ' for _ in range(len(str(self.iters)) - len(str(i+1)))])
                    s2 = ''.join([' ' for _ in range(len(str(self.lam_steps)) - len(str(step+1)))])
                    print(f"\rSearch step {s2}{step+1}/{self.lam_steps}, APG0 Iteration {s1}{i+1}/{self.iters}  ", end='')
                grad, loss = self.get_grad_loss(x, w, y)

                if self.linesearch:
                    L = self.backtrack_line_search(x, w, y, grad, loss, L)
                else:
                    # step size decay
                    # we take the reciprocal twice since L is applied as 1 / L
                    L = 1 / (1 / L * math.sqrt(1 - i / self.iters))

                # FISTA update
                v_old = v.clone()
                v = self.prox(w - 1 / L * grad, x, L)
                t_old = t
                t = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
                w = v + ((t_old - 1) / t) * (v - v_old)

            if self.numclasses == 2:
                succ = self.predict(x + v) * y > 0
            else:
                succ = self.predict(x + v).argmax(-1) == y.view(-1)
            succ = succ.view(-1)
            if self.lam_steps == 1:
                best_w = v.clone()
                best_norm = v.norm(p=0, dim=-1)
            # save the sparsest successful CF so far
            better = (best_norm > v.norm(p=0, dim=-1)) & succ
            best_norm[better] = v[better].norm(p=0, dim=-1)
            best_w[better] = v[better].clone()
            # increase weight of the classification loss if no CF was found
            # and decrease if one was found
            lam_ub[succ] = self.lam[succ]
            lam_lb[~succ] = self.lam[~succ]
            mask = lam_ub < 1e9
            self.lam[mask] = (lam_ub[mask] + lam_lb[mask]) / 2
            self.lam[~mask] *= 10
        
        if self.verbose:
            print('')
        return (x + best_w).detach()
    


class APG0_CFE_KDE(APG0_CFE):
    '''
    Class for computing counterfactual explanations using the APG0 algorithm with KDEs or GMMs.
    '''
    def __init__(self, model: nn.Module, mins: Tensor, maxs: Tensor, numclasses: int,
                 kdes: List[KDE], range_min: Tensor | None = None,
                 range_max: Tensor | None = None, L0: float = 0.1, lam0: float = 0.1,
                 c: float = 1.0, beta: float = 0.5, iters: int = 100,
                 max_i: int = 100, eta: float = 1.1, theta: float = 0.5,
                 prox: str = 'zero', linesearch: bool = False, lam_steps: int = 10,
                 verbose: bool = False, scale_model: bool = True) -> None:
        '''
        Initialize the APG0_CFE_KDE class.

        Arguments:
            model:        nn.Module, the model to be explained
            mins:         Tensor, minimum values for each feature used for
                          scaling the data to [0, 1]
            numclasses:   int, number of classes
            maxs:         Tensor, maximum values for each feature used for
                          scaling the data to [0, 1]
            kdes:         List[KDE], KDEs for all classes
            range_min:    Tensor | None, minimum values for the perturbations.
                          If None, the minimum values are set to 0.
            range_max:    Tensor | None, maximum values for the perturbations.
                          If None, the maximum values are set to 1.
            L0:           float, Lipschitz constant
            lam:          float, regularization parameter
            c:            float, constant term in the loss function
            beta:         float, parameter for the proximal operator
            iters:        int, number of iterations
            max_i:        int, maximum number of iterations in the backtracking line search
            eta:          float, parameter for the backtracking line search
            theta:        float, weight of the KDE loss
            prox:         str, proximal operator to use. Options: 'zero', 'half',
                          'twothirds', 'one', 'zero_fixed', where 'zero_fixed' 
                          projects to a 0-norm ball of radius beta.
            linesearch:   bool, whether to use line search
            lam_steps:    int, number of steps for the section search
            verbose:      bool, whether to print the current section search step
                          and iteration
            scale_model:  bool, set to true to scale the model output from [0, 1] to
                          [-1, 1] for binary classification. (set to false if the model
                          output is already in [-1, 1])
        '''
        super().__init__(model=model, mins=mins, maxs=maxs, numclasses=numclasses,
                         range_min=range_min, range_max=range_max, L0=L0, lam0=lam0,
                         c=c, beta=beta, iters=iters, max_i=max_i, eta=eta, prox=prox,
                         linesearch=linesearch, lam_steps=lam_steps, verbose=verbose,
                         scale_model=scale_model)
        self.kdes = kdes
        for kde in self.kdes:
            kde.to(self.device, self.dtype)
        self.theta = theta


    def loss(self, x: Tensor, w: Tensor, y: Tensor, active: Tensor | None = None) -> Tensor:
        '''
        Compute the L2 regularized loss with a KDE or GMM loss.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            w: Tensor of shape (n, d), the perturbation
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, 1)
        '''
        loss = super().loss(x, w, y, active)
        kdeloss = torch.empty_like(loss)
        mapidx = [i for i in range(self.numclasses)] if self.numclasses != 2 else [-1, 1]
        for i in range(self.numclasses):
            mask = (y == mapidx[i]).view(-1)
            if mask.any():
                kdeloss[mask] = self.kdes[i]((x[mask] + w[mask])\
                              * (self.maxs - self.mins) + self.mins).view(-1, 1)
        return loss - self.theta * kdeloss
    

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().to(device, dtype)
        for i in range(len(self.kdes)):
            self.kdes[i].to(device=device, dtype=dtype)
    


class APG0_CFE_kNN(APG0_CFE):
    '''
    Class for computing counterfactual explanations using APG0 with density gravity.
    '''
    def __init__(self, model: nn.Module, mins: Tensor, maxs: Tensor, numclasses: int,
                 trains: List[Tensor], range_min: Tensor | None = None,
                 range_max: Tensor | None = None, L0: float = 0.1, lam0: float = 0.1,
                 c: float = 1.0, beta: float = 0.5, iters: int = 100,
                 max_i: int = 100, eta: float = 1.1, theta: float = 0.5,
                 k: int = 3, prox: str = 'zero', linesearch: bool = False, lam_steps: int = 10,
                 verbose: bool = False, scale_model: bool = True) -> None:
        '''
        Initialize the APG0_CFE_KDE class.

        Arguments:
            model:        nn.Module, the model to be explained
            mins:         Tensor, minimum values for each feature used for
                          scaling the data to [0, 1]
            maxs:         Tensor, maximum values for each feature used for
                          scaling the data to [0, 1]
            numclasses:   int, number of classes
            trains:       List[Tensor], training data for all classes
            range_min:    Tensor | None, minimum values for the perturbations.
                          If None, the minimum values are set to 0.
            range_max:    Tensor | None, maximum values for the perturbations.
                          If None, the maximum values are set to 1.
            L0:           float, Lipschitz constant
            lam:          float, regularization parameter
            c:            float, constant term in the loss function
            beta:         float, parameter for the proximal operator
            iters:        int, number of iterations
            max_i:        int, maximum number of iterations in the backtracking line search
            eta:          float, parameter for the backtracking line search
            theta:        float, weight of the kNN loss
            k:            int, number of nearest neighbors
            prox:         str, proximal operator to use. Options: 'zero', 'half',
                          'twothirds', 'one', 'zero_fixed', where 'zero_fixed' 
                          projects to a 0-norm ball of radius beta.
            linesearch:   bool, whether to use line search
            lam_steps:    int, number of steps for the section search
            verbose:      bool, whether to print the current section search step
                          and iteration
            scale_model:  bool, set to true to scale the model output from [0, 1] to
                          [-1, 1] for binary classification. (set to false if the model
                          output is already in [-1, 1])
        '''
        super().__init__(model=model, mins=mins, maxs=maxs, numclasses=numclasses,
                         range_min=range_min, range_max=range_max, L0=L0, lam0=lam0,
                         c=c, beta=beta, iters=iters, max_i=max_i, eta=eta, prox=prox,
                         linesearch=linesearch, lam_steps=lam_steps, verbose=verbose,
                         scale_model=scale_model)
        
        self.trains = [((train.clone() - mins) / (maxs - mins)).view(train.size(0), -1).cpu() for train in trains]
        self.theta = theta
        self.k = k
        
        # compute the local densities for the points in each class
        self.nbs = [NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train.cpu()) for train in self.trains]
        self.ldens = []
        for i in range(len(self.trains)):
            dists = torch.from_numpy(self.nbs[i].kneighbors(self.trains[i].cpu())[0])\
                .to(device=self.device, dtype=self.dtype)
            self.ldens.append(k / dists.sum(dim=1))


    def get_g(self, x, y):
        '''
        Compute the the convex combination g of the k nearest neighbors.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, d)
        '''
        g = torch.empty_like(x)
        mapidx = [i for i in range(self.numclasses)] if self.numclasses != 2 else [-1, 1]
        for i in range(self.numclasses):
            mask = (y == mapidx[i]).view(-1)
            if mask.any():
                idxs = torch.from_numpy(self.nbs[i].kneighbors(x[mask].cpu())[1]).to(self.device)
                a = self.ldens[i][idxs] / self.ldens[i][idxs].sum(-1).view(-1, 1)
                g[mask] = (self.trains[i].to(self.device)[idxs] * a[..., None]).sum(dim=1)

        return g

    
    def loss(self, x: Tensor, w: Tensor, y: Tensor, active: Tensor | None = None) -> Tensor:
        '''
        Compute the L2 regularized loss with a density gravity loss.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            w: Tensor of shape (n, d), the perturbation
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, 1)
        '''
        if active is None:
            active = torch.ones_like(self.lam, dtype=torch.bool)
        loss = super().loss(x, w, y, active)
        return loss + self.theta * ((x + w - self.g[active.view(-1)]) ** 2).sum(-1, keepdim=True)
    

    def get_CFs(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Compute the counterfactuals using the APG0 algorithm.

        Arguments:
            x: Tensor of shape (n, d) expected to be in [0, 1],
               the original data
            y: Tensor of shape (n, 1) expected to be in {-1, 1},
               the target label
        Returns:
            Tensor of shape (n, d)
        '''
        assert x.min() >= 0 and x.max() <= 1, "Data is expected to be in [0, 1]"
        assert x.shape[0] == y.shape[0], "Data and target label are expected to have the same number of samples"
        assert len(y.shape) == 2 and y.shape[1] == 1, "Target label is expected to be a tensor of shape (n, 1)"
        if self.numclasses == 2:
            mask1 = y == 1
            mask2 = y == -1
            assert (mask1 | mask2).all(), "Classes are expected to be in {-1, 1}"
        else:
            mask = y.view(y.size(0), 1) == torch.arange(self.numclasses, device=self.device, dtype=y.dtype).view(1, -1)
            assert (mask.any(-1)).all(), "Classes are expected to be in {0, ..., numclasses-1}"

        self.g = self.get_g(x, y)
        return super().get_CFs(x, y)