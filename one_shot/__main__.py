from modal import Function

from . import _main, stub


def main():
    _main(Function.lookup(stub.name, "Dreambooth.tune").remote_gen)


def test():
    Function.lookup(stub.name, "Dreambooth.generate").remote(
        "ca4b1e40984e7cc6f23777963e9ae76e"
    )


test()
