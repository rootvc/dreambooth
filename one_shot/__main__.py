from modal import Function

from . import _main, stub


def main():
    _main(Function.lookup(stub.name, "Dreambooth.tune").remote_gen)


def test():
    Function.lookup(stub.name, "Dreambooth.generate").remote(
        "f09e2b714736a0a553d33448fd6d9ed5"
    )


test()
