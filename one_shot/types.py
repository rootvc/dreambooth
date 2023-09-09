from typing import ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
T = TypeVar("T", covariant=True)


class PretrainedModel(Protocol[P, T]):
    @classmethod
    def from_pretrained(cls, *args: P.args, **kwargs: P.kwargs) -> T:
        ...


M = TypeVar("M", bound=PretrainedModel)
