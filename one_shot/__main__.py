from modal import Function

from . import _main, stub

_main(Function.lookup(stub.name, "Dreambooth.tune").remote_gen)
