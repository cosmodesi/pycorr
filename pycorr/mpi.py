import numpy as np

from mpi4py import MPI
from pmesh.domain import GridND

from .utils import BaseClass
from . import utils


COMM_WORLD = MPI.COMM_WORLD


def enum(*sequential, **named):
    """Enumeration values to serve as status tags passed between processes."""
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def split_ranks(nranks, nranks_per_worker, include_all=False):
    """
    Divide the ranks into chunks, attempting to have `nranks_per_worker` ranks
    in each chunk. This removes the master (0) rank, such
    that `nranks - 1` ranks are available to be grouped

    Parameters
    ----------
    nranks : int
        The total number of ranks available

    nranks_per_worker : int
        The desired number of ranks per worker

    include_all : bool, optional
        if `True`, then do not force each group to have
        exactly `nranks_per_worker` ranks, instead including the remainder as well;
        default is `False`
    """
    available = list(range(1, nranks)) # available ranks to do work
    total = len(available)
    extra_ranks = total % nranks_per_worker

    if include_all:
        for i, chunk in enumerate(np.array_split(available, max(total//nranks_per_worker, 1))):
            yield i, list(chunk)
    else:
        for i in range(total//nranks_per_worker):
            yield i, available[i*nranks_per_worker:(i+1)*nranks_per_worker]

        i = total // nranks_per_worker
        if extra_ranks and extra_ranks >= nranks_per_worker//2:
            remove = extra_ranks % 2 # make it an even number
            ranks = available[-extra_ranks:]
            if remove: ranks = ranks[:-remove]
            if len(ranks):
                yield i+1, ranks


class MPITaskManager(BaseClass):
    """
    A MPI task manager that distributes tasks over a set of MPI processes,
    using a specified number of independent workers to compute each task.

    Given the specified number of independent workers (which compute
    tasks in parallel), the total number of available CPUs will be
    divided evenly.

    The main function is ``iterate`` which iterates through a set of tasks,
    distributing the tasks in parallel over the available ranks.
    """
    def __init__(self, nprocs_per_task=1, use_all_nprocs=False, mpicomm=MPI.COMM_WORLD):
        """
        Initialize MPITaskManager.

        Parameters
        ----------
        nprocs_per_task : int, optional
            The desired number of processes assigned to compute each task.

        mpicomm : MPI communicator, optional
            The global communicator that will be split so each worker
            has a subset of CPUs available; default is COMM_WORLD.

        use_all_nprocs : bool, optional
            If `True`, use all available CPUs, including the remainder
            if `nprocs_per_task` does not divide the total number of CPUs
            evenly; default is `False`.
        """
        self.nprocs_per_task = nprocs_per_task
        self.use_all_nprocs  = use_all_nprocs

        # the base communicator
        self.basecomm = mpicomm
        self.rank = self.basecomm.rank
        self.size = self.basecomm.size

        # need at least one
        if self.size == 1:
            raise ValueError('Need at least two processes to use a MPITaskManager')

        # communication tags
        self.tags = enum('READY', 'DONE', 'EXIT', 'START')

        # the task communicator
        self.mpicomm = None

        # store a MPI status
        self.status = MPI.Status()

    def __enter__(self):
        """
        Split the base communicator such that each task gets allocated
        the specified number of nranks to perform the task with.
        """
        self.self_worker_ranks = []
        color = 0
        total_ranks = 0
        nworkers = 0

        # split the ranks
        for i, ranks in split_ranks(self.size, self.nprocs_per_task, include_all=self.use_all_nprocs):
            if self.rank in ranks:
                color = i+1
                self.self_worker_ranks = ranks
            total_ranks += len(ranks)
            nworkers = nworkers + 1
        self.other_ranks = [rank for rank in range(self.size) if rank not in self.self_worker_ranks]

        self.workers = nworkers # store the total number of workers
        if self.rank == 0:
            self.log_info('Entering {} with {:d} workers.'.format(self.__class__.__name__,self.workers))

        # check for no workers!
        if self.workers == 0:
            raise ValueError('no pool workers available; try setting `use_all_nprocs` = True')

        leftover = (self.size - 1) - total_ranks
        if leftover and self.rank == 0:
            self.log_warning('with `nprocs_per_task` = {:d} and {:d} available rank(s), '\
                                '{:d} rank(s) will do no work'.format(self.nprocs_per_task, self.size-1, leftover))
            self.log_warning('set `use_all_nprocs=True` to use all available nranks')

        # crash if we only have one process or one worker
        if self.size <= self.workers:
            raise ValueError('only have {:d} ranks; need at least {:d} to use the desired %d workers'.format(self.size, self.workers+1, self.workers))

        # ranks that will do work have a nonzero color now
        self._valid_worker = color > 0

        # split the comm between the workers
        self.mpicomm = self.basecomm.Split(color, 0)

        return self

    def is_root(self):
        """
        Is the current process the root process?
        Root is responsible for distributing the tasks to the other available ranks.
        """
        return self.rank == 0

    def is_worker(self):
        """
        Is the current process a valid worker?
        Workers wait for instructions from the master.
        """
        try:
            return self._valid_worker
        except AttributeError:
            raise ValueError('workers are only defined when inside the ``with MPITaskManager()`` context')

    def _get_tasks(self):
        """Internal generator that yields the next available task from a worker."""

        if self.is_root():
            raise RuntimeError('Root rank mistakenly told to await tasks')

        # logging info
        if self.mpicomm.rank == 0:
            self.log_debug('worker master rank is {:d} on {} with {:d} processes available'.format(self.rank, MPI.Get_processor_name(), self.mpicomm.size))

        # continously loop and wait for instructions
        while True:
            args = None
            tag = -1

            # have the master rank of the subcomm ask for task and then broadcast
            if self.mpicomm.rank == 0:
                self.basecomm.send(None, dest=0, tag=self.tags.READY)
                args = self.basecomm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()

            # bcast to everyone in the worker subcomm
            args = self.mpicomm.bcast(args) # args is [task_number, task_value]
            tag = self.mpicomm.bcast(tag)

            # yield the task
            if tag == self.tags.START:

                # yield the task value
                yield args

                # wait for everyone in task group before telling master this task is done
                self.mpicomm.Barrier()
                if self.mpicomm.rank == 0:
                    self.basecomm.send([args[0], None], dest=0, tag=self.tags.DONE)

            # see ya later
            elif tag == self.tags.EXIT:
                break

        # wait for everyone in task group and exit
        self.mpicomm.Barrier()
        if self.mpicomm.rank == 0:
            self.basecomm.send(None, dest=0, tag=self.tags.EXIT)

        # debug logging
        self.log_debug('rank %d process is done waiting',self.rank)

    def _distribute_tasks(self, tasks):
        """Internal function that distributes the tasks from the root to the workers."""

        if not self.is_root():
            raise ValueError('only the root rank should distribute the tasks')

        ntasks = len(tasks)
        task_index     = 0
        closed_workers = 0

        # logging info
        self.log_debug('master starting with {:d} worker(s) with {:d} total tasks'.format(self.workers, ntasks))

        # loop until all workers have finished with no more tasks
        while closed_workers < self.workers:

            # look for tags from the workers
            data = self.basecomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()

            # worker is ready, so send it a task
            if tag == self.tags.READY:

                # still more tasks to compute
                if task_index < ntasks:
                    this_task = [task_index, tasks[task_index]]
                    self.basecomm.send(this_task, dest=source, tag=self.tags.START)
                    self.log_debug('sending task `{}` to worker {:d}'.format(str(tasks[task_index]),source))
                    task_index += 1

                # all tasks sent -- tell worker to exit
                else:
                    self.basecomm.send(None, dest=source, tag=self.tags.EXIT)

            # store the results from finished tasks
            elif tag == self.tags.DONE:
                self.log_debug('received result from worker {:d}'.format(source))

            # track workers that exited
            elif tag == self.tags.EXIT:
                closed_workers += 1
                self.log_debug('worker {:d} has exited, closed workers = {:d}'.format(source,closed_workers))

    def iterate(self, tasks):
        """
        Iterate through a series of tasks in parallel.

        Notes
        -----
        This is a collective operation and should be called by all ranks.

        Parameters
        ----------
        tasks : iterable
            An iterable of `task` items that will be yielded in parallel
            across all ranks.

        Yields
        -------
        task :
            The individual items of `tasks`, iterated through in parallel.
        """
        # master distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():
            for tasknum, args in self._get_tasks():
                yield args

    def map(self, function, tasks):
        """
        Apply a function to all of the values in a list and return the list of results.

        If ``tasks`` contains tuples, the arguments are passed to
        ``function`` using the ``*args`` syntax.

        Notes
        -----
        This is a collective operation and should be called by
        all ranks.

        Parameters
        ----------
        function : callable
            The function to apply to the list.
        tasks : list
            The list of tasks.

        Returns
        -------
        results : list
            The list of the return values of ``function``.
        """
        results = []

        # master distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():

            # iterate through tasks in parallel
            for tasknum, args in self._get_tasks():

                # make function arguments consistent with *args
                if not isinstance(args, tuple):
                    args = (args,)

                # compute the result (only worker root needs to save)
                result = function(*args)
                if self.mpicomm.rank == 0:
                    results.append((tasknum, result))

        # put the results in the correct order
        results = self.basecomm.allgather(results)
        results = [item for sublist in results for item in sublist]
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit gracefully by closing and freeing the MPI-related variables."""

        if exc_value is not None:
            utils.exception_handler(exc_type, exc_value, exc_traceback)

        # wait and exit
        self.log_debug('Rank {:d} process finished'.format(self.rank))
        self.basecomm.Barrier()

        if self.is_root():
            self.log_debug('Master is finished; terminating')

        if self.mpicomm is not None:
            self.mpicomm.Free()


def gather_array(data, root=0, mpicomm=COMM_WORLD):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Gather the input data array from all ranks to the specified ``root``.
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like
        The data on each rank to gather.

    root : int, Ellipsis, default=0
        The rank number to gather the data to. If root is Ellipsis or None,
        broadcast the result to all ranks.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like, None
        The gathered data on root, and `None` otherwise.
    """
    if root is None: root = Ellipsis

    if np.isscalar(data):
        if root is Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=root)
        if mpicomm.rank == root:
            return np.array(gathered)
        return None

    if not isinstance(data, np.ndarray):
        raise ValueError('`data` must be numpy array in gather_array')

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather_array')

        # compute the new shape for each rank
        newlength = mpicomm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if root is Ellipsis or mpicomm.rank == root:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather_array(data[name], root=root, mpicomm=mpicomm)
            if root is Ellipsis or mpicomm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather_array')

    # check for bad dtypes and bad shapes
    if root is Ellipsis or mpicomm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None

    if root is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype), root=root)

    if bad_shape:
        raise ValueError('mismatch between shape[1:] across ranks in gather_array')
    if bad_dtype:
        raise ValueError('mismatch between dtypes across ranks in gather_array')

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = mpicomm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if root is Ellipsis or mpicomm.rank == root:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer


def broadcast_array(data, root=0, mpicomm=COMM_WORLD):
    """
    Broadcast the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to broadcast.

    root : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """

    # check for bad input
    if mpicomm.rank == root:
        isscalar = np.isscalar(data)
    else:
        isscalar = None
    isscalar = mpicomm.bcast(isscalar, root=root)

    if isscalar:
        return mpicomm.bcast(data, root=root)

    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input,root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in broadcast_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in broadcast_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
        np_dtype = np.dtype((dtype, shape))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape, 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # the return array
    recvbuffer = np.empty(shape, dtype=dtype, order='C')

    # the send offsets
    counts = np.ones(mpicomm.size, dtype='i', order='C')
    offsets = np.zeros_like(counts, order='C')

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


def local_size(size, mpicomm=COMM_WORLD):
    """
    Divide global ``size`` into local (process) size.

    Parameters
    ----------
    size : int
        Global size.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    localsize : int
        Local size. Sum of local sizes over all processes equals global size.
    """
    start = mpicomm.rank * size // mpicomm.size
    stop = (mpicomm.rank + 1) * size // mpicomm.size
    localsize = stop - start
    #localsize = size // mpicomm.size
    #if mpicomm.rank < size % mpicomm.size: localsize += 1
    return localsize


def scatter_array(data, counts=None, root=0, mpicomm=COMM_WORLD):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to split and scatter.

    counts : list of int
        List of the lengths of data to send to each rank.

    root : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != mpicomm.size:
            raise ValueError('counts array has wrong length!')

    # check for bad input
    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in scatter_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newshape[0] = newlength = local_size(shape[0], mpicomm=mpicomm)
    else:
        if counts.sum() != shape[0]:
            raise ValueError('the sum of the `counts` array needs to be equal to data length')
        newshape[0] = counts[mpicomm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = mpicomm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


def domain_decompose(mpicomm, smoothing, positions1, weights1=None, positions2=None, weights2=None, boxsize=None, domain_factor=None):
    """
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/pair_counters/domain.py.
    Decompose positions and weights on a grid of MPI processes.
    Requires mpi4py and pmesh.

    Parameters
    ----------
    mpicomm : MPI communicator
        The MPI communicator.

    smoothing : float
        The maximum Cartesian separation implied by the user's binning.

    positions1 : list, array
        Positions in the first catalog. Typically of shape (3, N) or (2, N);
        in the latter case input arrays are assumed to be angular coordinates in degrees
        (these will be projected onto the unit sphere for further decomposition).

    positions2 : list, array, default=None
        Optionally, for cross-pair counts, positions in the second catalog. See ``positions1``.

    weights1 : list, array, default=None
        Optionally, weights of the first catalog.

    weights2 : list, array, default=None
        Optionally, weights in the second catalog.

    boxsize : array, default=None
        For periodic wrapping, the 3 side-lengths of the periodic cube.

    domain_factor : int, default=None
        Multiply the size of the MPI mesh by this factor.
        If ``None``, defaults to 2 in case ``boxsize`` is ``None``,
        else (periodic wrapping) 1.

    Returns
    -------
    (positions1, weights1), (positions2, weights2) : arrays
        The (decomposed) set of positions and weights.
    """
    if mpicomm.size == 1:
        return (positions1, weights1), (positions2, weights2)

    def split_size_3d(s):
        """
        Split `s` into three integers, a, b, c, such
        that a * b * c == s and a <= b <= c.
        """
        a = int(s ** (1./3.)) + 1
        d = s
        while a > 1:
            if s % a == 0:
                s = s // a
                break
            a = a - 1
        b = int(s ** 0.5) + 1
        while b > 1:
            if s % b == 0:
                s = s // b
                break
            b = b - 1
        c = s
        return a, b, c

    periodic = boxsize is not None
    angular = len(positions1) == 2
    ngrid = split_size_3d(mpicomm.size)
    if domain_factor is None:
        domain_factor = 1 if periodic else 2
    ngrid *= domain_factor

    size1 = mpicomm.allreduce(len(positions1[0]))

    auto = positions2 is None
    if auto:
        positions2 = positions1
        weights2 = weights1
        size2 = size1
    else:
        size2 = mpicomm.allreduce(len(positions2[0]))

    cpositions1 = positions1
    cpositions2 = positions2
    if angular:
        # project to unit sphere
        cpositions1 = utils.sky_to_cartesian([*positions1, np.ones_like(positions1[0])], degree=True)
        if auto:
            cpositions2 = cpositions1
        else:
            cpositions2 = utils.sky_to_cartesian([*positions2, np.ones_like(positions2[0])], degree=True)

    cpositions1 = np.array(cpositions1).T
    if periodic:
        cpositions1 %= boxsize
    if auto:
        cpositions2 = cpositions1
    else:
        cpositions2 = np.array(cpositions2).T
        if periodic:
            cpositions2 %= boxsize

    if periodic:
        posmin = np.zeros_like(boxsize)
        posmax = np.asarray(boxsize)
    else:
        def get_boxsize(positions):
            posmin, posmax = positions.min(axis=0), positions.max(axis=0)
            posmin = np.min(mpicomm.allgather(posmin), axis=0)
            posmax = np.max(mpicomm.allgather(posmax), axis=0)
            return posmin, posmax

        posmin, posmax = get_boxsize(cpositions1)
        if not auto:
            posmin2, posmax2 = get_boxsize(cpositions2)
            posmin = np.min([posmin, posmin2], axis=0)
            posmax = np.max([posmax, posmax2], axis=0)
        diff = max(np.abs(posmax - posmin).max(), 1.)
        posmin -= 1e-6 * diff # margin to make sure all positions will be included
        posmax += 1e-6 * diff

    # domain decomposition
    grid = [np.linspace(pmin, pmax, grid + 1, endpoint=True) for pmin, pmax, grid in zip(posmin, posmax, ngrid)]
    domain = GridND(grid, comm=mpicomm, periodic=periodic) # raises VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences

    if not periodic:
        # balance the load
        domain.loadbalance(domain.load(cpositions1))

    # exchange first particles
    layout = domain.decompose(cpositions1, smoothing=0)
    positions1 = layout.exchange(*positions1, pack=False) # exchange takes a list of arrays
    if weights1 is not None and len(weights1):
        multiple_weights = len(weights1) > 1
        weights1 = layout.exchange(*weights1, pack=False)
        if multiple_weights: weights1 = list(weights1)
        else: weights1 = [weights1]

    boxsize = posmax - posmin

    # exchange second particles
    if smoothing > boxsize.max() * 0.25:
        positions2 = [gather_array(p, root=Ellipsis, mpicomm=mpicomm) for p in positions2]
        if weights2 is not None: weights2 = [gather_array(w, root=Ellipsis, mpicomm=mpicomm) for w in weights2]
    else:
        layout = domain.decompose(cpositions2, smoothing=smoothing)
        positions2 = layout.exchange(*positions2, pack=False)
        if weights2 is not None and len(weights2):
            multiple_weights = len(weights2) > 1
            weights2 = layout.exchange(*weights2, pack=False)
            if multiple_weights: weights2 = list(weights2)
            else: weights2 = [weights2]

    nsize1 = mpicomm.allreduce(len(positions1[0]))
    assert nsize1 == size1, 'some particles1 disappeared (after: {:d} v.s. before: {:d})...'.format(nsize1, size1)

    positions1 = list(positions1) # exchange returns tuple
    positions2 = list(positions2)
    return (positions1, weights1), (positions2, weights2)
