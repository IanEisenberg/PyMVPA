# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dynamic Mode Decomposition (DMD)"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug


import numpy as np
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples


class DMD(Mapper):
    """Dynamic Mode Decomposition (DMD)

    TODO: DOC and Params


    """
    def __init__(self):
        """
        Parameters

        """
        Mapper.__init__(self)
        # Spatial modes
        self.phi = None
        # Temporal modes?
        self.A_evals = None
        # Fourier spectrum of modes
        self.mu = None
        # Weight matrix, which can be estimated with regularization
        self.z = None

    @accepts_dataset_as_samples
    def _train(self, samples, r=None, dt=1.0):
        X = samples[:, :-1]
        Y = samples[:, 1:]
        [U, S, Vh] = np.linalg.svd(X)
        if r is None:
            r = np.rank(X)
        # A_tilda
        A_tilda = U[:, :r].T.dot(Y).dot(Vh[:r, :].T)/S[:r]
        # Eigen decomposition of A_tilda
        self.A_evals, W = np.linalg.eig(A_tilda)
        self.mu = np.log(self.A_evals)/dt
        self.phi = Y.dot(Vh[:r, :].T).dot(W) / S[:r]
        # Solving for z with first sample
        self.z = np.linalg.pinv(self.phi).dot(X[:, 0])


    def forward1(self, nsamples):
        """
        Forwarding/Reconstructing data using the DMD modes.

        :param data:
        :return:
        """
        # Initialize with zeros?
        Xhat = np.zeros(shape=(self.z.shape[0], nsamples), dtype=float)
        for tr in range(nsamples):
            Xhat = self.z.dot()
