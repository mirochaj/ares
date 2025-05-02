import gc

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    try:
        import dill
        MPI._p_pickle.dumps = dill.dumps
        MPI._p_pickle.loads = dill.loads
    except ImportError:
        pass
    except AttributeError: # named differently depending on version
        MPI.pickle.__init__(dill.dumps, dill.loads)
except ImportError:
    MPI = None
    rank = 0
    size = 1

try:
    import pymp
    have_pymp = True
except ImportError:
    have_pymp = False


class MPIPool(object): # pragma: no cover

    def __init__(self, comm=None, master=0):
        """
        Initialize an MPIPool object.

        Parameters
        ----------
        comm : mpi4py.MPI.COMM_WORLD instance.
            If None, one will be created.
        master : int
            ID # of root processor.

        """
        self.comm = MPI.COMM_WORLD if comm is None else comm

        assert self.comm.size > 1
        assert 0 <= master < self.comm.size

        self.master = master
        self.workers = set(range(self.comm.size))
        self.workers.discard(self.master)

    def is_master(self):
        return self.master == self.comm.rank

    def is_worker(self):
        return self.comm.rank in self.workers

    def map(self, function, iterable):
        assert self.is_master()

        comm = self.comm
        workerset = self.workers.copy()
        tasklist = [(tid, (function, arg)) for tid, arg in enumerate(iterable)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:

            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                comm.send(task, dest=worker, tag=taskid)

            if tasklist:
                flag = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag:
                    continue
            else:
                comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                status=status)
            worker = status.source
            workerset.add(worker)
            taskid = status.tag
            resultlist[taskid] = result
            pending -= 1

        return resultlist

    def start(self):
        if not self.is_worker():
            return

        comm = self.comm
        master = self.master
        status = MPI.Status()

        while True:
            task = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
            if task is None:
                break

            function, arg = task
            result = function(arg)
            comm.ssend(result, master, status.tag)

            del result, arg
            gc.collect()

    def stop(self):
        if not self.is_master():
            return
        for worker in self.workers:
            self.comm.send(None, worker, 0)

class WorkerPool(object):
    def __init__(self, nthreads=None):
        self.nthreads = nthreads

    @property
    def pool(self):
        if not hasattr(self, '_pool_'):
            assert have_pymp, "Need pymp if nthreads != 0 or None."
            pymp.config.num_threads = self.nthreads
            self._pool_ = pymp.Parallel(self.nthreads)
            self._pool_.__enter__()
            if self.is_pymp_pool and self.thread_num == 1:
                print(f"* Initialized pymp worker pool with {self.nthreads} threads")
        return self._pool_

    def done(self, exc_t=None, exc_val=None, exc_tb=None):
        if self.is_pymp_pool:
            self.pool.__exit__(exc_t, exc_val, exc_tb)
        else:
            pass

    @property
    def is_pymp_pool(self):
        if not hasattr(self, '_is_pymp_pool'):
            if self.nthreads not in [0, 1, None]:
                self._is_pymp_pool = True
            else:
                self._is_pymp_pool = False

        return self._is_pymp_pool

    @property
    def thread_num(self):
        if not hasattr(self, '_thread_num'):
            if self.is_pymp_pool:
                self._thread_num = self.pool.thread_num
            else:
                self._thread_num = 1

        return self._thread_num

    def xrange(self, N):
        if self.is_pymp_pool:
            return self.pool.xrange(0, N)
        else:
            return range(0, N)

    def get_buffer(self, shape, dtype):
        if self.is_pymp_pool:
            return pymp.shared.array(shape, dtype=dtype)
        else:
            return np.zeros(shape, dtype=dtype)
#if __name__ == '__main__':
#
#    pool = Pool(MPI.COMM_WORLD)
#
#    def sq(x): return x*x
#
#    pool.start()
#
#    if pool.is_master():
#
#        tic = MPI.Wtime()
#        res = list(pool.map(sq, range(100)))
#        toc = MPI.Wtime()
#
#        for y, x in zip(res, range(100)):
#            assert y ==  sq(x)
#
#    pool.stop()
