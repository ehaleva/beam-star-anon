import numpy as np

class GlobalRunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0                 # pooled M2 (within + between) — unchanged
        self.max_val = -np.inf
        self.kth_val = -np.inf
        self.min_max_val = np.inf
        # --- within-request accumulators (new) ---
        self.within_M2 = 0.0          # sum over requests of each request's own sum-sq-dev
        self.within_n  = 0            # sum over requests of (pool size)  [use n_i, see note]

    def push(self, array):
        x = np.asanyarray(array).ravel()
        batch_n = x.size
        if batch_n == 0:
            return

        batch_max = np.max(x)
        if batch_max > self.max_val:
            self.max_val = batch_max

        batch_mean = np.mean(x)
        batch_M2 = np.sum((x - batch_mean)**2)     # this request's OWN scatter, about its OWN mean

        # --- within-request pool: accumulate batch_M2 WITHOUT the delta (between) term ---
        self.within_M2 += batch_M2
        self.within_n  += batch_n                  # or += (batch_n - 1) for unbiased; see note

        # --- pooled (unchanged: Chan merge, keeps between term) ---
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.mean += delta * batch_n / total_n
            self.M2 += batch_M2 + (delta**2) * (self.n * batch_n / total_n)
            self.n = total_n

    @property
    def variance(self):                            # pooled = within + between (unchanged)
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    # --- new properties ---
    @property
    def within_variance(self):                     # sigma_within^2
        return self.within_M2 / self.within_n if self.within_n > 0 else 0.0

    @property
    def std_within(self):
        return np.sqrt(self.within_variance)

    @property
    def between_variance(self):                    # sigma_between^2 = pooled - within
        return max(self.variance - self.within_variance, 0.0)

    @property
    def std_between(self):
        return np.sqrt(self.between_variance)

    @property
    def max(self):
        return self.max_val if self.n > 0 else None