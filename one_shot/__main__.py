from modal import Function

from . import _main, stub


def main():
    _main(Function.lookup(stub.name, "Dreambooth.tune").remote_gen)


def test():
    Function.lookup(stub.name, "Dreambooth.generate").remote(
        "79c053cf2d6d3922be669f9b78f34b2a"
    )


test()
