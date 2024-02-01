class CInstruPair(object):
    def __init__(self, instru_a: str, instru_b: str):
        self.instru_a, self.instru_b = instru_a, instru_b

    @property
    def Id(self) -> str:
        return f"{self.instru_a}_{self.instru_b}"
