from clwl_experiments_module13_formal_comparison_runner import (
    default_mnist_real_config,
    save_suite_outputs,
)

cfg = default_mnist_real_config()
cfg.groups = ["g3_clpl_vs_clwl_order_preserving"]
files = save_suite_outputs(cfg)
print(files)