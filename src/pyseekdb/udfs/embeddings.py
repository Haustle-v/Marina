# SPDX-License-Identifier: PROPRIETARY
# SPDX-FileCopyrightText: Copyright The Geneva Authors

"""Pre-built embedding UDF helpers."""

import abc
import logging
import urllib.parse
from functools import cached_property
from typing import Any, List, Optional, Tuple, Dict

import attrs
import pyarrow as pa

from pyseekdb.transformer import UDF, udf

_LOG = logging.getLogger(__name__)

SENTENCE_TRANSFORMERS_FAMILY = "sentence-transformers"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_TRANSFORMER_COLUMN = "text"


def _extract_string_inputs(
    batch: pa.RecordBatch, column: str
) -> Tuple[List[str], List[int], List[Any]]:
    """Return valid string values and their row indices for ``column``."""

    index = batch.schema.get_field_index(column)
    if index == -1:
        raise ValueError(f"Column '{column}' not found in RecordBatch")

    field = batch.schema.field(index)
    if not pa.types.is_string(field.type) and not pa.types.is_large_string(field.type):
        raise TypeError(f"Column '{column}' must contain string data")

    values = batch.column(index).to_pylist()
    valid_indices: List[int] = []
    valid_texts: List[str] = []
    for idx, value in enumerate(values):
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"embedding UDF expects string inputs, received {type(value).__name__}."
            )
        valid_indices.append(idx)
        valid_texts.append(value)

    return valid_texts, valid_indices, values


def _resolve_device(num_gpus: float) -> Optional[str]:
    if num_gpus <= 0:
        return None

    try:
        import torch
    except ImportError:
        _LOG.warning("torch not available; falling back to CPU for embeddings")
        return None

    if torch.cuda.is_available():
        _LOG.debug("CUDA is available; using GPU for embeddings")
        return "cuda"

    _LOG.debug("GPU requested but CUDA is not available; using CPU")
    return None


@attrs.define
class _EmbeddingModel(abc.ABC):
    """
    Base class interface for implementing pre-baked embedding models.
    All model families are required to only implement:
    * _build_model - load and return the model instance
    * _get_dimension - return the model's embedding dimension
    * embed - embed a RecordBatch and return a ListArray of float32 embeddings

    The base class handles lazy model loading, dimension caching, and output type.

    Parameters
    ----------
    model_name:
        The model being used for embedding. It can be a name of a model
        to be loaded from HuggingFace Hub, a local path or in case of
        API-based models, it can be an endpoint URL or model ID.
    column:
        Name of the column that will be embedded.
    normalize:
        Whether to L2-normalise the generated embeddings.
    num_gpus:
        GPU allocation requested for the UDF. Values ``>= 0``
        positive values will request CUDA based execution.
    """

    model_name: str = attrs.field(kw_only=True)
    column: str = attrs.field(kw_only=True)
    normalize: bool = attrs.field(kw_only=True)
    num_gpus: float = attrs.field(
        default=0.0, kw_only=True, validator=attrs.validators.ge(0)
    )

    _device: Optional[str] = attrs.field(init=False, default=None)

    @abc.abstractmethod
    def _build_model(self) -> Any:
        """
        Return the model instance for the embedding backend. This
        method is called by UDF worker to lazily load the model.
        This can be used for one-time setup of the model instance, or
        client in case of API-based models (e.g. OpenAI, Gemini).
        """

    @abc.abstractmethod
    def _get_dimension(self) -> int:
        """
        Return the embedding dimension for the model.
        Embedding dimension must be a positive integer. It is the
        length of the embedding vector returned by the model for each input.
        """

    @abc.abstractmethod
    def embed(self, batch: pa.RecordBatch) -> pa.Array:
        """
        Embed ``batch`` and return a fixed-size list array of floats.
        It should handle missing inputs gracefully. For example::

            ["hello", None, "world"] -> [[0.1, 0.2], None, [0.3, 0.4]]
        """

    @cached_property
    def model(self) -> Any:
        """Lazily load and cache the model instance."""
        return self._build_model()

    @cached_property
    def dimension(self) -> int:
        """Lazily get and cache the model's embedding dimension."""
        dimension = self._get_dimension()
        if dimension <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        return dimension

    def output_type(self) -> pa.DataType:
        return pa.list_(pa.float32(), self.dimension)

    def __getstate__(self) -> Dict[str, Any]:
        """
        only serialize attributes excluding internal state or cached properties
        """
        return attrs.asdict(
            self,
            # include only the attributes that are part of __init__
            filter=lambda attribute, value: attribute.init,
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(**state)


@attrs.define
class _SentenceTransformersModel(_EmbeddingModel):
    trust_remote_code: bool = attrs.field(default=False, kw_only=True)

    def _build_model(self) -> Any:
        self._device = _resolve_device(self.num_gpus)
        model = self._load_model(self.model_name, self._device, self.trust_remote_code)
        return model

    def _get_dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        return int(dimension)

    def embed(self, batch: pa.RecordBatch) -> pa.Array:
        model = self.model
        valid_texts, valid_indices, values = _extract_string_inputs(batch, self.column)

        outputs: List[Optional[List[float]]] = [None] * len(values)
        if valid_texts:
            embeddings = model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            vectors = embeddings.tolist()
            for idx, vector in zip(valid_indices, vectors):
                outputs[idx] = vector

        return pa.array(outputs, type=self.output_type())

    @staticmethod
    def _load_model(
        model_name: str, device: Optional[str], trust_remote_code: bool
    ) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required; install via "
                "`pip install sentence-transformers`"
            ) from exc

        return SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )


def _build_embedding_udf(
    model: _EmbeddingModel,
    udf_name: str,
    num_gpus: float,
    dimension: Optional[int] = None,
) -> UDF:
    """
    Build an embedding UDF from a model.

    Parameters
    ----------
    model:
        The embedding model instance
    udf_name:
        Name for the UDF
    num_gpus:
        GPU allocation for the UDF
    dimension:
        Optional pre-specified embedding dimension. If None, will eagerly
        load the model to determine dimension. If provided, model loading
        is deferred until UDF execution.
    """
    if dimension is None:
        # Eager mode: load model now to get dimension
        dimension = model.dimension
        data_type = model.output_type()
    else:
        # Lazy mode: use provided dimension, defer model loading
        data_type = pa.list_(pa.float32(), dimension)

    @udf(
        name=udf_name,
        data_type=data_type,
        num_gpus=num_gpus,
    )
    class EmbeddingUDF:
        def __init__(self) -> None:
            self._model = model
            self.dimension = dimension

        def __call__(self, batch: pa.RecordBatch) -> pa.Array:
            return self._model.embed(batch)

    return EmbeddingUDF()  # type: ignore


# Supported pre-baked embedding UDFs


def sentence_transformer_udf(
    model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    column: str = DEFAULT_SENTENCE_TRANSFORMER_COLUMN,
    normalize: bool = True,
    num_gpus: float = 0.0,
    trust_remote_code: bool = False,
    dimension: Optional[int] = None,
) -> UDF:
    """
    Return a stateful sentence-transformers embedding UDF.

    Parameters
    ----------
    model:
        The model being used for embedding. by default, it uses
        ``sentence-transformers/all-MiniLM-L6-v2`` from HuggingFace Hub.
    column:
        Name of the column that will be embedded. By default, it uses ``text``.
    normalize:
        Whether to L2-normalise the generated embeddings. Defaults to ``True``.
    num_gpus:
        Fractional GPU allocation requested for the UDF. Values ``>= 0``
        Be default, keeps execution on CPU; positive values request CUDA.
    trust_remote_code:
        Whether to trust remote code when loading the model. Defaults to ``False``
        as recommended by sentence-transformers.
    dimension:
        Optional pre-specified embedding dimension. If None (default), will eagerly
        load the model to determine dimension. If provided, model loading is deferred
        until UDF execution. Use this for lazy loading when the model is not available
        at UDF definition time (e.g., in manifest upload scripts).

    Returns
    -------
    UDF
        A UDF instance that can be registered with a Geneva dataset.
    """

    embedding_model = _SentenceTransformersModel(
        model_name=model,
        column=column,
        normalize=normalize,
        num_gpus=num_gpus,
        trust_remote_code=trust_remote_code,
    )
    model_name_sanitized = urllib.parse.quote_plus(model)
    udf_name = f"{SENTENCE_TRANSFORMERS_FAMILY}:{model_name_sanitized}"
    return _build_embedding_udf(
        model=embedding_model,
        udf_name=udf_name,
        num_gpus=num_gpus,
        dimension=dimension,
    )

