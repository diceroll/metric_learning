import copy

import numpy as np
from chainer import configuration, cuda, function, link
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.training import extension, extensions
from chainercv.evaluations import eval_semantic_segmentation


class ModifiedEvaluator(extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, iterator, target, label_names=None, output_dir=None, device=None):
        super(ModifiedEvaluator, self).__init__(iterator, target, device=device)
        self.label_names = label_names
        self.output_dir = output_dir

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            kwargs = {}
            kwargs['train'] = False
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays, **kwargs)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays, **kwargs)
                    else:
                        eval_func(in_arrays, **kwargs)

            summary.add(observation)

        observation = summary.compute_mean()

        return observation
