import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy as np
from chainer import link
from chainer import reporter
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy


class CosineSimilarity(chainer.Chain):

    def __init__(self, n_input, n_class):
        super(CosineSimilarity, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(n_input, n_class, nobias=True)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x /= F.sqrt(F.batch_l2_norm_squared(x)).reshape((-1, 1))
        h = self.fc(x)
        h /= F.sqrt(F.sum(F.square(self.fc.W), axis=1))

        return h


class MetricLearnClassifier(L.Classifier):

    compute_accuracy = True

    def __init__(self, predictor, n_hidden, n_class,
                 method='arcface', final_margin=0.5, final_scale=64, target_epoch=None,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 train=True):
        super(MetricLearnClassifier, self).__init__(predictor)
        self.method = method
        self.final_margin = final_margin
        self.final_scale = final_scale
        self.target_epoch = target_epoch
        if target_epoch is not None:
            self.margin = final_margin
            self.scale = final_scale
        else:
            self.margin = 0
            self.scale = 1
        self.train = train
        with self.init_scope():
            self.cosine_similarity = CosineSimilarity(n_hidden, n_class)

    def forward(self, *args, **kwargs):

        if 'train' in kwargs.keys():
            self.train = kwargs['train']
            del kwargs['train']

        if 'epoch' in kwargs.keys():
            if self.target_epoch is not None:
                self.margin = kwargs['epoch'] / self.target_epoch * self.final_margin
                self.scale = 1 + kwargs['epoch'] / self.target_epoch * (self.final_scale - 1)
            del kwargs['epoch']

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.hidden_feature = None
        self.loss = None
        self.accuracy = None

        self.hidden_feature = self.predictor(*args, **kwargs)
        self.y = self.cosine_similarity(self.hidden_feature)

        if self.train:
            xp = chainer.backend.cuda.get_array_module(self.y)
            if self.method == 'sphereface':
                penalty = xp.zeros_like(self.y)
                rows = xp.arange(t.size)
                penalty[rows, t] = (F.cos(F.arccos(self.y[rows, t]) * self.margin) - self.y[rows, t]).data
                self.y += penalty
            elif self.method == 'arcface':
                penalty = xp.zeros_like(self.y)
                rows = xp.arange(t.size)
                penalty[rows, t] = (F.cos(F.arccos(self.y[rows, t]) + self.margin) - self.y[rows, t]).data
                self.y += penalty
            elif self.method == 'cosface':
                self.y[xp.arange(t.size), t] -= self.margin
            else:
                raise NotImplementedError
        self.y *= self.scale

        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
