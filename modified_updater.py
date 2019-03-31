from chainer.training import StandardUpdater


class ModifiedUpdater(StandardUpdater):

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        kwargs = {}
        kwargs['epoch'] = self.epoch

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays, **kwargs)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays, **kwargs)
        else:
            optimizer.update(loss_func, in_arrays, **kwargs)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
