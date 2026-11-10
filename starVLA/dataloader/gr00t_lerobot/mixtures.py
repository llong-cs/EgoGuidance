"""
mixtures.py

Base registry of named dataset mixtures.

A mixture maps a ``data_mix`` name (``datasets.<block>.data_mix``) to a list of
``(dataset_name, sampling_weight, robot_type)`` tuples, where ``dataset_name`` is a
sub-directory of ``datasets.<block>.data_root_dir`` and ``robot_type`` is a key of
``ROBOT_TYPE_CONFIG_MAP``.

The mixtures used in this repository are declared next to their benchmark example in
``examples/<Benchmark>/train_files/data_registry/data_config.py`` (``libero_all``,
``droid_success``, ``libero_all``, ...) and merged into this base registry by
:mod:`starVLA.dataloader.gr00t_lerobot.registry`. Add a new benchmark the same way.
"""

# {mixture_name: [(dataset_name, sampling_weight, robot_type), ...]}
DATASET_NAMED_MIXTURES: dict = {}
