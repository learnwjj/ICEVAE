from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
import random

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h




class MLP_Regression(nn.Module):
    def __init__(
        self,
        feature_dim,
        outcome_dist="bernoulli",
        latent_dim=20,
        hidden_dim=200,
        num_layers=3,
        num_samples=100,
    ):
        config = dict(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )
        for name, size in config.items():
            if not (isinstance(size, int) and size > 0):
                raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)

    def fit(
        self,
        x,
        t,
        y,
        num_epochs=100,
        batch_size=100,
        learning_rate=1e-3,
        learning_rate_decay=0.1,
        weight_decay=1e-4,
        log_every=100,
    ):
        """
        Train using :class:`~pyro.infer.svi.SVI` with the
        :class:`TraceCausalEffect_ELBO` loss.

        :param ~torch.Tensor x:
        :param ~torch.Tensor t:
        :param ~torch.Tensor y:
        :param int num_epochs: Number of training epochs. Defaults to 100.
        :param int batch_size: Batch size. Defaults to 100.
        :param float learning_rate: Learning rate. Defaults to 1e-3.
        :param float learning_rate_decay: Learning rate decay over all epochs;
            the per-step decay rate will depend on batch size and number of epochs
            such that the initial learning rate will be ``learning_rate`` and the final
            learning rate will be ``learning_rate * learning_rate_decay``.
            Defaults to 0.1.
        :param float weight_decay: Weight decay. Defaults to 1e-4.
        :param int log_every: Log loss each this-many steps. If zero,
            do not log loss. Defaults to 100.
        :return: list of epoch losses
        """
        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        self.whiten = PreWhitener(x)

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam(
            {
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "lrd": learning_rate_decay ** (1 / num_steps),
            }
        )
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())
        losses = []
        for epoch in range(num_epochs):
            for x, t, y in dataloader:
                x = self.whiten(x)
                loss = svi.step(x, t, y, size=len(dataset)) / len(dataset)
                if log_every and len(losses) % log_every == 0:
                    logger.debug(
                        "step {: >5d} loss = {:0.6g}".format(len(losses), loss)
                    )
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def ite(self, x, num_samples=None, batch_size=None):
        r"""
        Computes Individual Treatment Effect for a batch of data ``x``.

        .. math::

            ITE(x) = \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1) \bigr]
                   - \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0) \bigr]

        This has complexity ``O(len(x) * num_samples ** 2)``.

        :param ~torch.Tensor x: A batch of data.
        :param int num_samples: The number of monte carlo samples.
            Defaults to ``self.num_samples`` which defaults to ``100``.
        :param int batch_size: Batch size. Defaults to ``len(x)``.
        :return: A ``len(x)``-sized tensor of estimated effects.
        :rtype: ~torch.Tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logger.info("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        for x in dataloader:
            x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x)
            ite = (y1 - y0).mean(0)
            if not torch._C._get_tracing_state():
                logger.debug("batch ate = {:0.6g}".format(ite.mean()))
            result.append(ite)
        return torch.cat(result)

    def to_script_module(self):
        """
        Compile this module using :func:`torch.jit.trace_module` ,
        assuming self has already been fit to data.

        :return: A traced version of self with an :meth:`ite` method.
        :rtype: torch.jit.ScriptModule
        """
        self.train(False)
        fake_x = torch.randn(2, self.feature_dim)
        with pyro.validation_enabled(False):
            # Disable check_trace due to nondeterministic nodes.
            result = torch.jit.trace_module(self, {"ite": (fake_x,)}, check_trace=False)
        return result
