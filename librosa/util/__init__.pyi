from . import decorators
from . import exceptions

from .files import (
    find_files as find_files,
    example as example,
    ex as ex,
    list_examples as list_examples,
    example_info as example_info,
    cite as cite,
)

from .matching import (
    match_intervals as match_intervals,
    match_events as match_events,
)

from .deprecation import (
    Deprecated as Deprecated,
    rename_kw as rename_kw,
)

from ._nnls import (
    nnls as nnls,
)

from .utils import (
    MAX_MEM_BLOCK as MAX_MEM_BLOCK,
    frame as frame,
    pad_center as pad_center,
    expand_to as expand_to,
    fix_length as fix_length,
    valid_audio as valid_audio,
    valid_int as valid_int,
    is_positive_int as is_positive_int,
    valid_intervals as valid_intervals,
    fix_frames as fix_frames,
    axis_sort as axis_sort,
    localmax as localmax,
    localmin as localmin,
    normalize as normalize,
    peak_pick as peak_pick,
    sparsify_rows as sparsify_rows,
    shear as shear,
    stack as stack,
    fill_off_diagonal as fill_off_diagonal,
    index_to_slice as index_to_slice,
    sync as sync,
    softmask as softmask,
    buf_to_float as buf_to_float,
    tiny as tiny,
    cyclic_gradient as cyclic_gradient,
    dtype_r2c as dtype_r2c,
    dtype_c2r as dtype_c2r,
    count_unique as count_unique,
    is_unique as is_unique,
    abs2 as abs2,
    phasor as phasor,
)
