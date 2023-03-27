class Annotator:
    def __init__(self, verbose: bool = False, device: str | None = None):
        self.verbose = verbose
        self.device = device

    def is_valid(self) -> bool:
        raise NotImplementedError

    def annotate(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return NotImplementedError
