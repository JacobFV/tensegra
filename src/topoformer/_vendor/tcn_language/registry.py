"""Restricted registry for the vendored three-lesson closure."""
from .lessons import LESSON_CLASSES

def get(lesson_id):
    return {c.id: c for c in LESSON_CLASSES}[lesson_id]()

def all_lessons(*, implemented_only=False):
    return {c.id: c() for c in sorted(LESSON_CLASSES, key=lambda c: c.id)}
