from functools import cache
from typing import get_args as get_type_args  # noqa: F401 (better name)

from inflection import camelize, pluralize, singularize, underscore


@cache
def modelize(table_name: str) -> str:
    """Inverse of tabelize(...)"""
    return camelize(cached_singularize(table_name), uppercase_first_letter=True)


@cache
def cached_underscore(string: str) -> str:
    return underscore(string)


@cache
def cached_singularize(word: str) -> str:
    return singularize(word)


@cache
def cached_pluralize(word: str) -> str:
    return pluralize(word)


def is_plural(word: str) -> bool:
    return word == pluralize(word)


def inflect(word: str, count: int) -> str:
    """
    Pluralize or Singularize word based on count.
    """
    return singularize(word) if count == 1 else pluralize(word)
