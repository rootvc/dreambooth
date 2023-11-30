from pydantic import BaseModel


class PromptSegment(BaseModel):
    raw: str
    modifier: float

    @classmethod
    def new(cls, raw: str, modifier: float):
        return cls(raw=raw, modifier=modifier)

    @classmethod
    def plus(cls, raw: str, count: int = 1):
        return cls(raw=raw, modifier=1.1**count)

    @classmethod
    def minus(cls, raw: str, count: int = 1):
        return cls(raw=raw, modifier=0.9**count)

    def __str__(self):
        if self.modifier:
            return f"({self.raw}){self.modifier:.2f}"
        else:
            return self.raw


class PromptStrings(BaseModel):
    positives: list[str | PromptSegment]
    negatives: list[str | PromptSegment]

    def positive(self, **kwargs) -> str:
        return ", ".join(map(str, self.positives)).format(**kwargs)

    def negative(self, **kwargs) -> str:
        return ", ".join(map(str, self.negatives)).format(**kwargs)


class PromptTemplates(BaseModel):
    background: PromptStrings | None = None
    eyes: PromptStrings
    merge: PromptStrings
    details: PromptStrings


F = PromptSegment.new
P = PromptSegment.plus
M = PromptSegment.minus
