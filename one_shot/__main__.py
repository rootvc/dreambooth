from modal import Function

from . import _main, stub


def main():
    _main(Function.lookup(stub.name, "Dreambooth.tune").remote_gen)


def test():
    Function.lookup(stub.name, "Dreambooth.generate").remote(
        "d41d8cd98f00b204e9800998ecf8427e"
    )


test()
