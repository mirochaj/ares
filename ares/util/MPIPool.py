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
