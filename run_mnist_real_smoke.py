from clwl_experiments_module13_formal_comparison_runner import (
    default_mnist_real_config,
    save_suite_outputs,
)

cfg = default_mnist_real_config()
cfg.seeds = [0, 1, 2]
files = save_suite_outputs(cfg)
print(files)