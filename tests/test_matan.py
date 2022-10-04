from typing import Literal, Union, overload
import numpy as np
np.count_nonzero([[1,2,3],[4,5,6]])

from numpy.typing import ArrayLike
a: ArrayLike = [[1,2,3],[4,5,6]]
np.count_nonzero(a)


from numbers import Real, Number

x: Number = 123.2
y: Real = np.float_(123)

x: Number = np.int64(123)

x: np.number = 123

from librosa.util import abs2

abs2(3)

x: complex = 1


@overload
def test(x: Literal[False]) -> None: ...
@overload
def test(x: Literal[True]) -> int: ...
def test(x: bool) -> Union[int, None]:
    if x:
        return 1
    return None

a1 = test(True)
a2 = test(False)
b: bool = get_bool()
a3 = test(b)


from typing import overload, Union, Literal

@overload
def myfunc(arg: Literal[True]) -> str: ...

@overload
def myfunc(arg: Literal[False]) -> int: ...

def myfunc(arg: bool) -> Union[int, str]:
    if arg: return "something"
    else: return 0

a = bool("asf")
myfunc(a)