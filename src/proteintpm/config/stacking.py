# Number of nearest target atoms retained per environment (padded/truncated).
# This is the fixed tensor width fed to the model; environments are ranked by
# distance to the shard centre, so truncation keeps the k nearest.
broom_max_neighbours: int = 100

# Window used by the density validity filter (distinct from broom_max_neighbours):
# only the this-many nearest neighbours are considered when testing enclosure.
broom_filter_window: int = 25

# "Near" shell: at least `broom_min_count_low` neighbours (within the filter
# window) must lie within `broom_lower_threshold` angstrom of the shard centre.
broom_lower_threshold: float = 8.0  # angstrom
broom_min_count_low: int = 5

# "Far" shell: at most `broom_max_count_high` neighbours (within the filter
# window) may lie beyond `broom_upper_threshold` angstrom of the shard centre.
broom_upper_threshold: float = 13.0  # angstrom
broom_max_count_high: int = 3

# Reserved atom class for padding rows. Real atom classes begin at 1
# (the +1 offset applied in features/atom_types.map_atom), so class 0 is
# unambiguously padding.
pad_class: int = 0
