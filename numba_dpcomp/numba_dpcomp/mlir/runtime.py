# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import atexit
from numba.np.ufunc.parallel import get_thread_count
from .utils import load_lib, register_cfunc

runtime_lib = load_lib("dpcomp-runtime")

_init_func = runtime_lib.dpcompParallelInit
_init_func.argtypes = [ctypes.c_int]
_init_func(get_thread_count())

_finalize_func = runtime_lib.dpcompParallelFinalize

_funcs = [
    "dpcompParallelFor",
    "memrefCopy",
    "dpcompTakeContext",
    "dpcompReleaseContext",
    "dpcompPurgeContext",
]

for name in _funcs:
    func = getattr(runtime_lib, name)
    register_cfunc(name, func)


@atexit.register
def _cleanup():
    _finalize_func()
