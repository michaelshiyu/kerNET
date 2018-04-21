class _ensemble():
    def add(self,):
        raise NotImplementedError
    def forward(self,):
        raise NotImplementedError


class _kerLinearEnsemble():
    def add(self, component):
        assert isinstance(component, kerLinear)

    def forward(self, x, X):
        i = 0
        dims = [(
            getattr('comp'+str(i)).ker_dim,
            getattr('comp'+str(i)).out_dim
            ) for i in range(self._comp_counter)] # TODO: is this necessary?

        assert # TODO: out_dims of all comps should be equal
        out_dim = dims[0][1]
        y = Variable(torch.FloatTensor(x.shape[0], out_dim))
        if x.is_cuda: y=y.cuda()
        for i in range(self._comp_counter):
            # TODO: should probably check \sum_i comp_i.ker_dim == X.shape[0]
            # and perhaps also comp_i.ker_dim == comp_j.ker_dim unless j or i is
            # -1 (the last comp)
            component = getattr('comp'+str(i))
            X_batch = next(K.get_batch(X, batch_size=component.ker_dim))[0]
            # TODO: add to document of K.get_batch: whenever the argument has
            # only 1 set, add [0] to the return value
            y.add(component.forward(x, X_batch))

def baseMLKN.to_ensemble(self, layer, X_batch_size):
    assert isinstance(layer, kerLinear)
    ensemble = _kerLinearEnsemble()
    n_batch = layer.ker_dim // X_batch_size
    last_batch = layer.ker_dim % X_batch_size

    for i in range(n_batch):
        ensemble.add(kerLinear(
            ker_dim=batch_size,
            out_dim=layer.out_dim,
            sigma=layer.sigma,
            bias=layer.bias
            ))
        ensemble._comp_counter += 1
    if last_batch:
        ensemble.add(kerLinear(
            ker_dim=last_batch,
            out_dim=layer.out_dim,
            sigma=layer.sigma,
            bias=layer.bias
            ))
        ensemble._comp_counter += 1
