import numpy as np

def bootstrap(data, n, axis=0, func=np.var, func_kwargs={"ddof" : 1}):
    """Produce n bootstrap samples of data of the statistic given by func.
    
    Arguments
    ---------
    data : numpy.ndarray
        Data to resample.
    n : int
        Number of bootstrap trails.
    axis : int, optional
        Axis along which to resample. (Default ``0``).
    func : callable, optional
        Statistic to calculate. (Default ``numpy.var``).
    func_kwargs : dict, optional
        Dictionary with extra arguments for func. (Default ``{"ddof" : 1}``).
        
    Returns
    -------
    samples : numpy.ndarray
        Bootstrap samples of statistic func on the data.
    """
    
    if axis != 0:
        raise NotImplementedError("Only axis == 0 supported.")

    fiducial_output = func(data, axis=axis, **func_kwargs)
    
    if isinstance(data, list):
        assert all([d.shape[1:] == data[0].shape[1:] for d in data])
    
    samples = np.zeros((n, *fiducial_output.shape), dtype=fiducial_output.dtype)

    for i in range(n):
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True) for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)], axis=axis, **func_kwargs)
        else:
            idx = np.random.choice(data.shape[axis], size=data.shape[axis], replace=True)
            samples[i] = func(data[idx], axis=axis, **func_kwargs)

    return samples

def bootstrap_var(data, n, axis=0, func=np.var, func_kwargs={"ddof" : 1}):
    """Calculate the variance of the statistic given by func.
    
    Arguments
    ---------
    data : numpy.ndarray
        Data to resample.
    n : int
        Number of bootstrap trails.
    axis : int, optional
        Axis along which to resample. (Default ``0``).
    func : callable, optional
        Statistic to calculate. (Default ``numpy.var``).
    func_kwargs : dict, optional
        Dictionary with extra arguments for func. (Default ``{"ddof" : 1}``).
        
    Returns
    -------
    var : numpy.ndarray
        Bootstrap variance of statistic func on the data.
    """
    samples = bootstrap(data, n, axis, func, func_kwargs)
    return samples.var(axis=axis, ddof=1)