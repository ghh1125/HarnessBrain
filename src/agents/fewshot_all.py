
from ..llm import LLMCallable
from .fewshot_memory import FewShotMemory


class FewShotAll(FewShotMemory):

    def __init__(self, llm: LLMCallable):
        super().__init__(llm, max_examples=9999)
