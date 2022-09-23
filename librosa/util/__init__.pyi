from . import decorators
from . import exceptions

from .files import find_files, example, ex, list_examples, example_info

from .matching import match_intervals, match_events

from .deprecation import (
    Deprecated,
    rename_kw,
)

from ._nnls import nnls

from .utils import (
    MAX_MEM_BLOCK,
    frame,
    pad_center,
    expand_to,
    fix_length,
    valid_audio,
    valid_int,
    is_positive_int,
    valid_intervals,
    fix_frames,
    axis_sort,
    localmax,
    localmin,
    normalize,
    peak_pick,
    sparsify_rows,
    shear,
    stack,
    fill_off_diagonal,
    index_to_slice,
    sync,
    softmask,
    buf_to_float,
    tiny,
    cyclic_gradient,
    dtype_r2c,
    dtype_c2r,
    count_unique,
    is_unique,
    abs2,
    phasor,
)
