from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs

class FactMetric(object):
    """Class that maintains exact match metrics for facts in FVQA dialog."""

    def __init__(self, opt):
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics['correct'] = 0
        if opt.get('numthreads', 1) > 1:
            self.metrics = SharedTable(self.metrics)
        self.datatype = opt.get('datatype', 'train')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def __str__(self):
        return str(self.metrics)

    def __repr__(self):
        return repr(self.metrics)

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return self

    def update(self, observation, labels):
        with self._lock():
            self.metrics['cnt'] += 1

        # Exact match metric.
        correct = 0
        prediction = observation.get('text', None)
        if prediction == labels[0]:
            correct = 1
        with self._lock():
            self.metrics['correct'] += correct

        # Return a dict containing the metrics for this specific example.
        # Metrics across all data is stored internally in the class, and
        # can be accessed with the report method.
        loss = {}
        loss['correct'] = correct
        return loss

    def report(self):
        # Report the metric over all data seen so far.
        m = {}
        m['total'] = self.metrics['cnt']
        if self.metrics['cnt'] > 0:
            m['accuracy'] = round_sigfigs(
                self.metrics['correct'] / self.metrics['cnt'], 4)
        return m

    def clear(self):
        with self._lock():
            self.metrics['cnt'] = 0
            self.metrics['correct'] = 0
