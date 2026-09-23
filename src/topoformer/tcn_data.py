"""Pinned TCN constructions, compiled once and rendered across public surfaces.

Only TCNSurface crosses the actor boundary. Dataset constructors and privileged
records are supervision/evaluation utilities, never language parsers for actors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .semantic_graph import COMPILER_VERSION, SemanticGraph, compile_term

SOURCE_COMMIT = '018c9ce9286fa4961292fe1e024b49fcd7e2dd7f'
RENDERER_VERSION = 'tcn-fixed-construction-v1'
LESSONS = ('variable_binding', 'unification', 'set_operations')
LANGUAGES = ('english', 'spanish', 'symbols')


@dataclass(frozen=True)
class TCNSurface:
    language: str
    text: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class TCNPrivileged:
    graph: SemanticGraph
    answer: str
    metadata_json: str


@dataclass(frozen=True)
class TCNExample:
    public: tuple[TCNSurface, ...]
    privileged: TCNPrivileged
    audit: dict[str, Any]


def _vendor_path() -> Path:
    return Path(__file__).parent / '_vendor' / 'tcn_language'


def verify_vendor_manifest() -> bool:
    manifest = json.loads((_vendor_path() / 'MANIFEST.json').read_text())
    return manifest['source_commit'] == SOURCE_COMMIT and all(
        hashlib.sha256((_vendor_path() / name).read_bytes()).hexdigest() == entry['vendored_sha256']
        for name, entry in manifest['files'].items())


def build_tcn_example(lesson: str, seed: int, *, languages: Sequence[str] = LANGUAGES,
                      difficulty: float | None = None) -> TCNExample:
    """Generate once in fixed English context, then render the same Term object.

    The supported lessons have identity-valued choices, unchanged by rendering.
    Unknown lessons/languages fail closed; no derived language database is used.
    """
    if lesson not in LESSONS:
        raise ValueError(f'Unsupported lesson: {lesson}')
    aliases = {'en': 'english', 'es': 'spanish'}
    language_codes = tuple(aliases.get(code, code) for code in languages)
    if not language_codes or any(code not in LANGUAGES for code in language_codes):
        raise ValueError('Expected english, spanish, or symbols surfaces')
    from ._vendor.tcn_language._structure import to_json
    from ._vendor.tcn_language.context import GenerationContext
    from ._vendor.tcn_language.generators.extra import ACTIVE_LANGUAGE
    from ._vendor.tcn_language.languages import get_language
    from ._vendor.tcn_language.lessons.variable_binding import VariableBinding
    from ._vendor.tcn_language.lessons.unification import Unification
    from ._vendor.tcn_language.lessons.set_operations import SetOperations

    constructors = dict(zip(LESSONS, (VariableBinding, Unification, SetOperations)))
    token = ACTIVE_LANGUAGE.set('english')
    try:
        term, choices, answer, hidden = constructors[lesson]().build(
            seed, GenerationContext(language='english', difficulty=difficulty))
    finally:
        ACTIVE_LANGUAGE.reset(token)
    graph = compile_term(to_json(term))
    options = tuple(str(c) for c in choices)
    public = tuple(TCNSurface(code, get_language(code).render(term), options)
                   for code in language_codes)
    source_manifest = (_vendor_path() / 'MANIFEST.json').read_bytes()
    audit = {'lesson': lesson, 'seed': seed, 'difficulty': difficulty,
             'source_commit': SOURCE_COMMIT, 'compiler_version': COMPILER_VERSION,
             'renderer_version': RENDERER_VERSION, 'semantic_digest': graph.digest(),
             'source_manifest_sha256': hashlib.sha256(source_manifest).hexdigest(),
             'compiler_sha256': hashlib.sha256(Path(__file__).with_name('semantic_graph.py').read_bytes()).hexdigest(),
             'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'surface_sha256': {s.language: hashlib.sha256(s.text.encode()).hexdigest() for s in public},
             'distinct_surface_texts': len({s.text for s in public}),
             'execution_scope': 'semantic_compiler_only'}
    return TCNExample(public, TCNPrivileged(graph, str(answer), json.dumps(hidden,sort_keys=True)), audit)


def build_tcn_corpus(*, lessons: Sequence[str] = LESSONS, count: int = 100,
                     seed: int = 0, languages: Sequence[str] = LANGUAGES,
                     difficulty: float | None = None) -> tuple[TCNExample, ...]:
    """Deterministic balanced lesson cycle; count is total semantic constructions."""
    if count < 0 or not lessons:
        raise ValueError('count must be nonnegative and lessons nonempty')
    return tuple(build_tcn_example(lessons[i % len(lessons)], seed + i,
                                  languages=languages, difficulty=difficulty)
                 for i in range(count))
