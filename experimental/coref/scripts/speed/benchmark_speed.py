from typing import Callable, List
import torch
import typer
import timeit
import traceback
from pathlib import Path
import logging
from wasabi import msg

from data_reader import read_data, rebatch_texts
from logger import create_logger
from spacy.util import minibatch

DEFAULT_BATCH_SIZE = 256


def main(
    txt_dir: Path,
    result_dir: Path,
    library,
    name: str,
    gpu: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_texts: int = 0,
):
    try:
        data = read_data(txt_dir, limit=n_texts)
        articles = len(data)
        if articles == 0:
            msg.fail(
                f"Could not read any data from {txt_dir}: make sure a corpus of .txt files is available."
            )
        chars = sum([len(d) for d in data])
        words = sum([len(d.split()) for d in data])

        nlp_function = _get_run(library, name, gpu)
        start = timeit.default_timer()

        nlp_function(data, batch_size)
        end = timeit.default_timer()

        log_run = create_logger(result_dir)
        s = end - start
        log_run(
            library=library,
            name=name,
            gpu=gpu,
            articles=articles,
            characters=chars,
            words=words,
            seconds=s,
        )
    # Usually we avoid these kind of long try-except blocks, but here we just want to ensure
    # that the script can continue benchmarking the speed of other libraries if one fails
    except Exception as e:
        msg.info(f"Could not run model {name} with library {library} on GPU={gpu}:")
        msg.info(traceback.format_exc())


def _get_run(library: str, name: str, gpu: bool) -> Callable[[List[str]], None]:
    if library == "spacy":
        return _run_spacy_model(name, gpu)

    if library == "neuralcoref":
        return _run_neuralcoref_model(name, gpu)

    if library == "fastcoref":
        return _run_fastcoref_model(name, gpu)

    if library == "lingmesscoref":
        return _run_lingmesscoref_model(name, gpu)

    msg.fail(
        f"Can not parse models for library {library}. "
        f"Known libraries are: ['spacy', 'neuralcoref']. ",
        exits=1,
    )


def _run_spacy_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained spaCy pipeline"""
    import spacy

    if gpu:
        spacy.require_gpu(0)
    nlp = spacy.load(name)

    def run(texts: List[str], batch_size: int):
        list(nlp.pipe(texts, batch_size=batch_size))

    return run


def _run_neuralcoref_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    import spacy
    import neuralcoref

    if gpu:
        spacy.require_gpu()
    nlp = spacy.load("en")
    neuralcoref.add_to_pipe(nlp)

    def run(texts: List[str], batch_size: int):
        list(nlp.pipe(texts, batch_size=batch_size))

    return run


def _run_fastcoref_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    import spacy
    from fastcoref import spacy_component

    device = "cuda" if gpu else "cpu"
    nlp = spacy.load("en_core_web_sm")
    for pipe, _ in nlp.components:
        nlp.remove_pipe(pipe)

    nlp.add_pipe("fastcoref", config={"device": device})

    def run(texts: List[str], batch_size: int):
        list(nlp.pipe(texts, batch_size=batch_size))

    return run


def _run_lingmesscoref_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    import spacy
    from fastcoref import spacy_component

    device = "cuda" if gpu else "cpu"
    nlp = spacy.load("en_core_web_sm")
    for pipe, _ in nlp.components:
        nlp.remove_pipe(pipe)

    nlp.add_pipe(
        "fastcoref",
        config={
            "device": device,
            "model_architecture": "LingMessCoref",
            "model_path": "biu-nlp/lingmess-coref",
        },
    )

    def run(texts: List[str], batch_size: int):
        list(nlp.pipe(texts, batch_size=batch_size))

    return run


if __name__ == "__main__":
    typer.run(main)
