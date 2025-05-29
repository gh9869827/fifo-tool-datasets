import pytest
from fifo_tool_datasets.sdk.hf_dataset_adapters.sqna import (
    SQNAAdapter
)
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
)
import pathlib

def test_from_dat_to_wide_dataset():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_01.dat"
    dataset = adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]["in"] == "question 1"
    assert dataset[0]["out"] == "answer 1"
    assert dataset[1]["in"] == "question 2"
    assert dataset[1]["out"] == "answer 2"

def test_from_dat_to_wide_dataset_broken_01():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_broken_01.dat"

    with pytest.raises(SyntaxError, match="Invalid syntax \\(missing answer\\)"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_02():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_broken_02.dat"

    with pytest.raises(SyntaxError, match="Invalid syntax \\(missing question\\)"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_03():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_broken_03.dat"

    with pytest.raises(SyntaxError, match="Invalid syntax \\(missing question\\)"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_04():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_broken_04.dat"

    with pytest.raises(SyntaxError, match="Invalid syntax \\(missing answer\\)"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]
