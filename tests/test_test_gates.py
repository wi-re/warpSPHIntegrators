"""Reject missing and overlapping coverage before any numerical sweep starts."""

import pytest
from types import SimpleNamespace

from conftest import (CATEGORIES, CI_SHARDS, category_map, validate_shards,
                      pytest_collection_modifyitems)


def test_ci_shards_partition_every_category():
    validate_shards(CATEGORIES, CI_SHARDS)


@pytest.mark.parametrize('shards', [[], ['core', 'core or solvers']])
def test_missing_or_overlapping_shards_fail_collection(shards):
    with pytest.raises(pytest.UsageError, match='expected exactly one'):
        validate_shards({'core': ['test_example.py']}, shards)


def test_a_file_cannot_belong_to_two_categories():
    with pytest.raises(pytest.UsageError, match='multiple test categories'):
        category_map({'core': ['test_example.py'], 'solvers': ['test_example.py']})


def test_an_uncategorized_collected_file_is_rejected():
    item = SimpleNamespace(fspath=SimpleNamespace(basename='test_new_driver.py'))
    with pytest.raises(pytest.UsageError, match='lacks a test category'):
        pytest_collection_modifyitems(None, [item])
