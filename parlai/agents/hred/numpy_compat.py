'''
Compatibility with older numpy's providing argpartition replacement.

'''


'''
Created on Sep 12, 2014

@author: chorows
'''

__all__ = ['argpartition']

import numpy
import warnings

if hasattr(numpy, 'argpartition'):
    argpartition = numpy.argpartition
else:
    try:
        import bottleneck
        #warnings.warn('Your numpy is too old (You have %s, we need 1.7.1), but we have found argpartsort in bottleneck' % (numpy.__version__,))
        def argpartition(a, kth, axis=-1):
            return bottleneck.argpartsort(a, kth, axis)
    except ImportError:
        warnings.warn('''Beam search will be slow!

Your numpy is old (you have v. %s) and doesn't provide an argpartition function.
Either upgrade numpy, or install bottleneck (https://pypi.python.org/pypi/Bottleneck).

If you run this from within LISA lab you probably want to run: pip install bottleneck --user
''' % (numpy.__version__,))
        def argpartition(a, kth, axis=-1, order=None):
            return numpy.argsort(a, axis=axis, order=order)
