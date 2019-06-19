import numpy as np

import simple_bootstrap

def test_bootstrap_var():
    n_bootstrap = 1000
    
    # Create some random data
    n_data = 1000
    var_true = np.array((1.0, 0.5))
    data = np.random.normal(scale=np.sqrt(var_true), size=(n_data, 2))
    
    var_on_mean = simple_bootstrap.bootstrap.bootstrap_var(data, n_bootstrap, func=np.mean, func_kwargs={})
    
    print(f"True variance on the mean: {var_true[0]/n_data}, {var_true[1]/n_data}. Bootstrap: {var_on_mean}.")
    
if __name__ == "__main__":
    test_bootstrap_var()