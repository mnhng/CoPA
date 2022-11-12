from .dag import Diamond, TriagL, TriagR
from .data import SynGenerator, CmnistGenerator


class SynDia(Diamond, SynGenerator):
    pass


class SynTriL(TriagL, SynGenerator):
    pass


class SynTriR(TriagR, SynGenerator):
    pass


class CmnistDia(Diamond, CmnistGenerator):
    pass


class CmnistTriL(TriagL, CmnistGenerator):
    pass


class CmnistTriR(TriagR, CmnistGenerator):
    pass
