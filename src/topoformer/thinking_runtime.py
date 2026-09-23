"""Stage 6 incremental exact execution over protected immutable scalar registers.

The session accepts public literals and actor proposals, never a task, gold trace,
learned routing, or workspace. Rejections have no effect on protected memory.
"""
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

Scalar = int | float | bool
PRIMITIVE_NAMES = ('add', 'sub', 'mul', 'neg', 'compare')
VALUE_TYPES = ('integer', 'float', 'boolean')


@dataclass(frozen=True)
class ValueRegister:
    id: str
    value: Scalar
    type: str
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    id: str
    primitive: str
    arguments: tuple[str, ...]
    readiness: float = 1.0


@dataclass(frozen=True)
class ReturnEvent:
    candidate_id: str
    primitive: str
    arguments: tuple[str, ...]
    value: Scalar | None = None
    type: str | None = None
    register_id: str | None = None
    provenance: tuple[str, ...] = ()
    status: str = 'rejected'
    reason: str = ''
    affected: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrimitiveSchema:
    """P/name, Sigma/signature, V/validity, T/transition, and L/U boundaries.

    L/U are explicit external adapters; neither has access to session memory.
    Only the fixed T implementation below can append protected registers.
    """
    name: str
    inputs: tuple[str, ...]
    output: str
    validity: str = 'current finite numeric registers; bounded result; bool excluded'
    transition: str = 'append immutable result register with ordered provenance'

    def lower(self, latent, lowering: Callable):
        return lowering(latent, self)

    def lift(self, event: ReturnEvent, lifting: Callable):
        return lifting(event, self)


PRIMITIVES: Mapping[str, PrimitiveSchema] = MappingProxyType({
    name: PrimitiveSchema(name, ('numeric',) if name == 'neg' else ('numeric', 'numeric'),
                          'boolean' if name == 'compare' else 'numeric')
    for name in PRIMITIVE_NAMES
})


def _identifier(value) -> bool:
    return type(value) is str and bool(value) and len(value) <= 256


def _provenance(value) -> bool:
    return type(value) is tuple and all(_identifier(x) for x in value)


def _scalar_type(value: Scalar) -> str:
    return {int: 'integer', float: 'float', bool: 'boolean'}[type(value)]


class ProtectedSession:
    """Append-only scalar memory, with snapshot isolation per execute call.

    Same-ID distinct proposals conflict symmetrically. Shared reads are safe;
    there are no mutable targets. Repeated completed identities are idempotent.
    Result identity is ``result:<candidate.id>`` and never aliases a literal.
    """

    def __init__(self, initial_values: Iterable[ValueRegister], *, max_abs_value: float = 1e12):
        if type(max_abs_value) not in (int, float) or not math.isfinite(max_abs_value) or max_abs_value <= 0:
            raise ValueError('max_abs_value must be finite and positive')
        self._max_abs_value = max_abs_value
        registers = {}
        for register in initial_values:
            if type(register) is not ValueRegister:
                raise TypeError('initial_values must contain only ValueRegister objects')
            if not _identifier(register.id) or not _provenance(register.provenance):
                raise ValueError('invalid register identity or provenance')
            if not self._valid_value(register.value) or register.type != _scalar_type(register.value):
                raise ValueError('invalid register type or bounded scalar value')
            if register.id in registers:
                raise ValueError('duplicate initial register identity')
            registers[register.id] = register
        self._registers = registers
        self._completed: dict[str, tuple[tuple, ReturnEvent]] = {}
        self._events: list[ReturnEvent] = []

    @property
    def registers(self) -> Mapping[str, ValueRegister]:
        return MappingProxyType(self._registers)

    @property
    def events(self) -> tuple[ReturnEvent, ...]:
        """Only committed events; refusals are returned to the caller for audit."""
        return tuple(self._events)

    def _valid_value(self, value) -> bool:
        if type(value) is bool:
            return True
        if type(value) not in (int, float):
            return False
        return abs(value) <= self._max_abs_value and (type(value) is int or math.isfinite(value))

    @staticmethod
    def _check_candidate(candidate):
        if type(candidate) is not Candidate:
            raise TypeError('proposals must contain only Candidate objects')
        if not _identifier(candidate.id) or not _identifier(candidate.primitive):
            raise ValueError('invalid candidate identity or primitive')
        if type(candidate.arguments) is not tuple or not all(_identifier(x) for x in candidate.arguments):
            raise ValueError('arguments must be an ordered tuple of register identities')

    def execute(self, proposals: Iterable[Candidate], threshold: float = .5) -> tuple[ReturnEvent, ...]:
        """Gate independently, validate on current state, then commit valid writes.

        Malformed API containers raise before any mutation. Well-shaped but
        invalid operations return refusals. The method never checks a gold action.
        """
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite in [0,1]')
        candidates = tuple(proposals)
        for candidate in candidates:
            self._check_candidate(candidate)
        snapshot = dict(self._registers)
        signatures: dict[str, set[tuple]] = {}
        for c in candidates:
            signatures.setdefault(c.id, set()).add((c.primitive, c.arguments))
        pending: dict[str, tuple[tuple, ReturnEvent]] = {}
        returned = []
        for c in candidates:
            event = ReturnEvent(c.id, c.primitive, c.arguments)
            signature = (c.primitive, c.arguments)
            previous = self._completed.get(c.id) or pending.get(c.id)
            if len(signatures[c.id]) > 1 or (previous is not None and previous[0] != signature):
                returned.append(replace(event, status='conflict', reason='candidate identity has incompatible proposals'))
                continue
            if type(c.readiness) not in (int, float) or not math.isfinite(c.readiness) or not 0 <= c.readiness <= 1:
                returned.append(replace(event, reason='readiness must be finite in [0,1]'))
                continue
            if c.readiness < threshold:
                returned.append(replace(event, status='deferred', reason='below readiness threshold'))
                continue
            if previous is not None:
                returned.append(replace(previous[1], status='duplicate', affected=()))
                continue
            result_id = 'result:' + c.id
            if result_id in snapshot:
                returned.append(replace(event, status='conflict', reason='result register already exists'))
                continue
            schema = PRIMITIVES.get(c.primitive)
            if schema is None or len(c.arguments) != len(schema.inputs):
                returned.append(replace(event, reason='unknown primitive or wrong arity'))
                continue
            args = [snapshot.get(key) for key in c.arguments]
            if any(arg is None or arg.type not in ('integer', 'float') for arg in args):
                returned.append(replace(event, reason='argument unavailable or not numeric'))
                continue
            values = [arg.value for arg in args]
            try:
                if c.primitive == 'add': value = values[0] + values[1]
                elif c.primitive == 'sub': value = values[0] - values[1]
                elif c.primitive == 'mul': value = values[0] * values[1]
                elif c.primitive == 'neg': value = -values[0]
                else: value = values[0] < values[1]
            except (OverflowError, ArithmeticError):
                returned.append(replace(event, reason='numeric overflow'))
                continue
            if not self._valid_value(value):
                returned.append(replace(event, reason='result exceeds finite numeric bound'))
                continue
            # Ordered unique ancestors plus transition identity; argument order
            # itself is always retained, including repeated arguments.
            provenance = tuple(dict.fromkeys(p for arg in args for p in (arg.provenance or (arg.id,)))) + ('candidate:' + c.id,)
            event = replace(event, value=value, type=_scalar_type(value), register_id=result_id,
                            provenance=provenance, status='executed', affected=(result_id,))
            pending[c.id] = (signature, event)
            returned.append(event)
        for key, (signature, event) in pending.items():
            self._registers[event.register_id] = ValueRegister(event.register_id, event.value, event.type, event.provenance)
            self._completed[key] = (signature, event)
            self._events.append(event)
        return tuple(returned)
