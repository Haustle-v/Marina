# SPDX-License-Identifier: PROPRIETARY
# SPDX-FileCopyrightText: Copyright The Geneva Authors

import enum
import functools
import hashlib
import inspect
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union

import attrs
import numpy
import pyarrow as pa
from attrs import validators as valid

import cloudpickle as pickle

_LOG = logging.getLogger(__name__)

# special column name used to mark the rows that were not selected
# for backfilling.  This is used to avoid calling expensive UDFs
# on rows that are not selected.
BACKFILL_SELECTED = "__geneva_backfill_selected"


class UDFArgType(enum.Enum):
    """
    The type of arguments that the UDF expects.
    """

    # Scalar Batch
    SCALAR = 0
    # Array mode
    ARRAY = 1
    # Pass a pyarrow RecordBatch
    RECORD_BATCH = 2


@attrs.define
class UDF:  # type: ignore
    """User-defined function (UDF) to be applied to a Lance Table."""

    # The reference to the callable
    func: Callable = attrs.field()
    name: str = attrs.field(default="")
    cuda: Optional[bool] = attrs.field(default=False)
    num_cpus: Optional[float] = attrs.field(
        default=1.0,
        converter=lambda v: None if v is None else float(v),
        validator=valid.optional(valid.ge(0.0)),
        on_setattr=[attrs.setters.convert, attrs.setters.validate],
    )
    num_gpus: Optional[float] = attrs.field(
        default=None,
        converter=lambda v: None if v is None else float(v),
        validator=valid.optional(valid.ge(0.0)),
        on_setattr=[attrs.setters.convert, attrs.setters.validate],
    )
    memory: Optional[int] = attrs.field(default=None)
    batch_size: Optional[int] = attrs.field(default=None)

    # Error handling configuration removed for now

    def _record_batch_input(self) -> bool:
        sig = inspect.signature(self.func)
        if len(sig.parameters) == 1:
            param = list(sig.parameters.values())[0]
            return param.annotation == pa.RecordBatch
        return False

    @property
    def arg_type(self) -> UDFArgType:
        if self._record_batch_input():
            return UDFArgType.RECORD_BATCH
        if _is_batched_func(self.func):
            return UDFArgType.ARRAY
        return UDFArgType.SCALAR

    input_columns: Optional[List[str]] = attrs.field(default=None)

    data_type: pa.DataType = attrs.field(default=None)

    version: str = attrs.field(default="")

    checkpoint_key: str = attrs.field(default="")

    field_metadata: Dict[str, str] = attrs.field(default={})

    def __attrs_post_init__(self) -> None:
        """
        Initialize UDF fields and normalize num_gpus after all fields are set:
          1) if cuda=True and num_gpus is None or 0.0 -> set to 1.0
          2) otherwise ignore cuda and just use num_gpus setting
        """
        # Set default name
        if not self.name:
            if inspect.isfunction(self.func):
                self.name = self.func.__name__
            elif isinstance(self.func, Callable):
                self.name = self.func.__class__.__name__
            else:
                raise ValueError(
                    f"func must be a function or a callable, got {self.func}"
                )

        # Set default input_columns
        if self.input_columns is None:
            sig = inspect.signature(self.func)
            params = list(sig.parameters.keys())
            if self._record_batch_input():
                self.input_columns = None
            else:
                self.input_columns = params

        # Validate input_columns
        if self.arg_type == UDFArgType.RECORD_BATCH:
            if self.input_columns is not None:
                raise ValueError(
                    "RecordBatch input UDF must not declare any input columns. "
                    "RecordBatch UDFs receive the entire batch and should not "
                    "specify input_columns. Consider using a stateful RecordBatch "
                    "UDF and parameterize it or use UDF with Array inputs."
                )
        else:
            if self.input_columns is None:
                raise ValueError("Array and Scalar input UDF must declare input column")

        # Set default data_type
        if self.data_type is None:
            if self.arg_type != UDFArgType.SCALAR:
                raise ValueError(
                    "batched UDFs do not support data_type inference yet,"
                    " please specify data_type",
                )
            self.data_type = _infer_func_arrow_type(self.func, None)  # type: ignore[arg-type]

        # Validate data_type
        if self.data_type is None:
            raise ValueError("data_type must be set")
        if not isinstance(self.data_type, pa.DataType):
            raise ValueError(
                f"data_type must be a pyarrow.DataType, got {self.data_type}"
            )

        # Set default version
        if not self.version:
            hasher = hashlib.md5()
            hasher.update(pickle.dumps(self.func))
            self.version = hasher.hexdigest()

        # Set default checkpoint_key
        if not self.checkpoint_key:
            self.checkpoint_key = f"{self.name}:{self.version}"

        # Handle cuda/num_gpus normalization
        if self.cuda:
            warnings.warn(
                "The 'cuda' flag is deprecated. Please set 'num_gpus' explicitly "
                "(0.0 for CPU, >=1.0 for GPU).",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.num_gpus is None:
            self.num_gpus = 1.0 if self.cuda is True else 0.0
        # otherwise fall back to user specified num_gpus

    def __repr__(self) -> str:
        """Custom repr that safely handles missing attributes during unpickling.

        This is necessary because attrs-generated __repr__ can fail when called
        during exception handling in Ray if the object hasn't been fully unpickled yet.
        """
        try:
            # Try to get all attrs fields safely
            field_strs = []
            for field in attrs.fields(self.__class__):
                # Check if attribute exists first before accessing it
                if hasattr(self, field.name):
                    value = getattr(self, field.name)
                    field_strs.append(f"{field.name}={value!r}")
                else:
                    field_strs.append(f"{field.name}=<not set>")

            return f"{self.__class__.__qualname__}({', '.join(field_strs)})"
        except Exception:
            # Fallback if even that fails
            return f"<{self.__class__.__name__} (repr failed)>"

    def _scalar_func_record_batch_call(self, record_batch: pa.RecordBatch) -> pa.Array:
        """
        We use this when the UDF uses single call like
        `func(x_int, y_string, ...) -> type`

        this function automatically dispatches rows to the func and returns `pa.Array`
        """

        # this let's us avoid having to allocate a list in python
        # to hold the results. PA will allocate once for us
        def _iter():  # noqa: ANN202
            batches = (
                record_batch.to_pylist()
                if isinstance(record_batch, pa.RecordBatch)
                else record_batch
            )
            for item in batches:
                # we know inputs_columns is not none here
                if BACKFILL_SELECTED not in item or item.get(BACKFILL_SELECTED):
                    # we know input_columns is not none here
                    args = [item[col] for col in self.input_columns]  # type: ignore
                    yield self.func(*args)
                else:
                    # item was not selected, so do not compute
                    yield None

        arr = pa.array(
            _iter(),
            type=self.data_type,
        )
        # this should always by an Array, never should we get a ChunkedArray back here
        assert isinstance(arr, pa.Array)
        return arr

    def _input_columns_validator(self, attribute, value) -> None:
        """Validate input_columns attribute for attrs compatibility."""
        if self.arg_type == UDFArgType.RECORD_BATCH:
            if value is not None:
                raise ValueError(
                    "RecordBatch input UDF must not declare any input columns. "
                    "RecordBatch UDFs receive the entire batch and should not "
                    "specify input_columns."
                )
        else:
            if value is None:
                raise ValueError("Array and Scalar input UDF must declare input column")

    def validate_against_schema(
        self, table_schema: pa.Schema, input_columns: Optional[List[str]] = None
    ) -> None:
        """
        Validate UDF against table schema.

        This is the primary validation method that should be called before executing
        a UDF. It performs comprehensive validation including:

        1. **Column Existence**: Verifies all input columns exist in the table schema
        2. **Type Compatibility**: Checks that column types match UDF type annotations
           (if present)
        3. **RecordBatch Constraints**: Ensures RecordBatch UDFs don't have
           input_columns defined

        The validation happens at two points in the UDF lifecycle:
        - At `add_columns()` time when defining the column
        - At `backfill()` time when executing (if input_columns are overridden)

        Parameters
        ----------
        table_schema: pa.Schema
            The schema of the table being processed
        input_columns: list[str] | None
            The input column names to validate. If None, uses self.input_columns.

        Raises
        ------
        ValueError: If validation fails for any of the following reasons:
            - Input columns don't exist in table schema
            - Type mismatch between table and UDF expectations
            - RecordBatch UDF has input_columns defined
            - Array/Scalar UDF has no input_columns defined

        Warns
        -----
        UserWarning: If type validation is skipped due to:
            - UDF has no type annotations
            - Type annotation can't be mapped to PyArrow types

        Examples
        --------
        >>> @udf(data_type=pa.int32())
        ... def my_udf(a: int) -> int:
        ...     return a * 2
        >>> my_udf.validate_against_schema(table.schema)  # Validates column 'a' exists
        """

        # Determine which columns to validate
        cols_to_validate = (
            input_columns if input_columns is not None else self.input_columns
        )

        # Check RecordBatch UDFs
        if self.arg_type == UDFArgType.RECORD_BATCH:
            # Error if input_columns are specified for RecordBatch UDFs
            if cols_to_validate is not None:
                raise ValueError(
                    f"UDF '{self.name}' is a RecordBatch UDF but has input_columns "
                    f"{cols_to_validate} specified. RecordBatch UDFs receive the "
                    f"entire batch and should not declare input_columns. "
                    f"Remove the input_columns parameter."
                )
            # RecordBatch UDFs don't need column validation
            return

        # For Array and Scalar UDFs, input_columns must be defined
        if cols_to_validate is None:
            arg_type_name = self.arg_type.name if self.arg_type else "UNKNOWN"
            raise ValueError(
                f"UDF '{self.name}' (type: {arg_type_name}) has no input_columns "
                f"defined. Array and Scalar UDFs must specify input columns either "
                f"through function parameter names or the input_columns parameter."
            )

        # Validate all input columns exist in table schema
        missing_columns = [
            col for col in cols_to_validate if col not in table_schema.names
        ]

        if missing_columns:
            raise ValueError(
                f"UDF '{self.name}' expects input columns {missing_columns} which are "
                f"not found in table schema. Available columns: {table_schema.names}. "
                f"Check your UDF's function parameter names or input_columns parameter."
            )

        # Validate type compatibility for each input column
        self._validate_column_types(table_schema, cols_to_validate)

    def _validate_column_types(
        self, table_schema: pa.Schema, input_columns: List[str]
    ) -> None:
        """
        Validate type compatibility between table schema and UDF expectations.

        This method checks if the table column types match the UDF's type annotations.
        If no type annotations are present or types can't be mapped, validation is
        skipped with a warning.

        Parameters
        ----------
        table_schema: pa.Schema
            The schema of the table being processed
        input_columns: list[str]
            The input column names to validate types for

        Raises
        ------
        ValueError: If there's a type mismatch between table schema and UDF expectations

        Warns
        -----
        UserWarning: If type validation is skipped due to missing annotations or
            unmappable types
        """
        import warnings

        # Get type annotations from the UDF function
        annotations = _get_annotations(self.func)

        if not annotations:
            # No type annotations found - warn user
            warnings.warn(
                f"UDF '{self.name}' has no type annotations. Type validation will be "
                f"skipped. Consider adding type hints to your UDF function parameters "
                f"for better error detection.",
                UserWarning,
                stacklevel=4,
            )
            return

        # For each input column, validate type if annotation exists
        for col_name in input_columns:
            # Get the actual type from table schema
            table_field = table_schema.field(col_name)
            table_type = table_field.type

            # Get expected type from UDF signature if available
            if col_name in annotations:
                expected_type = annotations[col_name]

                # Try to map expected type to PyArrow type for comparison
                try:
                    expected_pa_type = self._python_type_to_arrow_type(expected_type)

                    # Check if types are compatible
                    if not self._types_compatible(table_type, expected_pa_type):
                        raise ValueError(
                            f"Type mismatch for column '{col_name}' in UDF "
                            f"'{self.name}': table has type {table_type}, but UDF "
                            f"expects {expected_pa_type} (from annotation "
                            f"{expected_type}). This will likely cause serialization "
                            f"or conversion errors during execution."
                        )
                except (ValueError, KeyError):
                    # If we can't map the type, skip validation with warning
                    warnings.warn(
                        f"Could not validate type for column '{col_name}' in UDF "
                        f"'{self.name}' with annotation {expected_type}. Type "
                        f"validation skipped for this column.",
                        UserWarning,
                        stacklevel=4,
                    )

    def _python_type_to_arrow_type(self, python_type) -> pa.DataType:
        """
        Convert Python type annotation to PyArrow type.

        Raises ValueError if type cannot be mapped.
        """
        # Handle PyArrow types directly
        if isinstance(python_type, pa.DataType):
            return python_type

        # Handle pa.Array annotation (for batched UDFs)
        if python_type == pa.Array:
            # Can't determine specific array type, so return None to skip validation
            raise ValueError("Cannot validate generic pa.Array type")

        # Map Python/numpy types to PyArrow types
        type_map = {
            bool: pa.bool_(),
            bytes: pa.binary(),
            float: pa.float32(),
            int: pa.int64(),
            str: pa.string(),
            numpy.bool_: pa.bool_(),
            numpy.uint8: pa.uint8(),
            numpy.uint16: pa.uint16(),
            numpy.uint32: pa.uint32(),
            numpy.uint64: pa.uint64(),
            numpy.int8: pa.int8(),
            numpy.int16: pa.int16(),
            numpy.int32: pa.int32(),
            numpy.int64: pa.int64(),
            numpy.float16: pa.float16(),
            numpy.float32: pa.float32(),
            numpy.float64: pa.float64(),
            numpy.str_: pa.string(),
        }

        if python_type in type_map:
            return type_map[python_type]

        raise ValueError(f"Cannot map Python type {python_type} to PyArrow type")

    def _types_compatible(self, actual: pa.DataType, expected: pa.DataType) -> bool:
        """
        Check if actual type is compatible with expected type.

        This is more permissive than exact equality, allowing for:
        - Exact matches
        - Nullable vs non-nullable variants
        """
        # Exact match
        if actual == expected:
            return True

        # Check base types match (ignoring nullability, precision differences)
        # For numeric types, check if they're in the same family
        if pa.types.is_integer(actual) and pa.types.is_integer(expected):
            # Allow integer types if bit width and signedness match
            return actual.bit_width == expected.bit_width and (
                (
                    pa.types.is_signed_integer(actual)
                    and pa.types.is_signed_integer(expected)
                )
                or (
                    pa.types.is_unsigned_integer(actual)
                    and pa.types.is_unsigned_integer(expected)
                )
            )

        if pa.types.is_floating(actual) and pa.types.is_floating(expected):
            # Require exact match for floating point types (float32 vs float64 matters!)
            return actual.bit_width == expected.bit_width

        # For other types, require exact match
        return False

    def __call__(self, *args, use_applier: bool = False, **kwargs) -> pa.Array:
        # dispatch coming from Applier or user calling with a `RecordBatch`
        if use_applier or (len(args) == 1 and isinstance(args[0], pa.RecordBatch)):
            record_batch = args[0]
            if self.arg_type == UDFArgType.SCALAR:
                return self._scalar_func_record_batch_call(record_batch)
            elif self.arg_type == UDFArgType.ARRAY:
                # Validate columns exist before accessing them
                try:
                    arrs = [record_batch[col] for col in self.input_columns]  # type:ignore
                except KeyError as e:
                    raise KeyError(
                        f"UDF '{self.name}' failed: column {e} not found in "
                        f"RecordBatch. Available columns: "
                        f"{record_batch.schema.names}. UDF expects "
                        f"input_columns: {self.input_columns}."
                    ) from e
                return self.func(*arrs)
            elif self.arg_type == UDFArgType.RECORD_BATCH:
                if isinstance(record_batch, pa.RecordBatch):
                    return self.func(record_batch)
                # a list of dicts with BlobFiles that need to de-ref'ed
                assert isinstance(record_batch, list)
                rb_list = []
                for row in record_batch:
                    new_row = {}
                    for k, v in row.items():
                        # Removed BlobFile support as it is specific to Lance
                        new_row[k] = v
                    rb_list.append(new_row)

                rb = pa.RecordBatch.from_pylist(rb_list)

                return self.func(rb)
        # dispatch is trying to access the function's original pattern
        return self.func(*args, **kwargs)


def udf(
    func: Optional[Callable] = None,
    *,
    data_type: Optional[pa.DataType] = None,
    version: Optional[str] = None,
    cuda: bool = False,  # deprecated
    field_metadata: Optional[Dict[str, str]] = None,
    input_columns: Optional[List[str]] = None,
    num_cpus: Optional[Union[int, float]] = None,
    num_gpus: Optional[Union[int, float]] = None,
    # error_handling removed
    **kwargs,
) -> Union[UDF, functools.partial]:
    """Decorator of a User Defined Function ([UDF][pyseekdb.transformer.UDF]).

    Parameters
    ----------
    func: Callable
        The callable to be decorated. If None, returns a partial function.
    data_type: pa.DataType, optional
        The data type of the output PyArrow Array from the UDF.
        If None, it will be inferred from the function signature.
    version: str, optional
        A version string to manage the changes of function.
        If not provided, it will use the hash of the serialized function.
    cuda: bool, optional, Deprecated
        If true, load CUDA optimized kernels.  Equvalent to num_gpus=1
    field_metadata: dict[str, str], optional
        A dictionary of metadata to be attached to the output `pyarrow.Field`.
    input_columns: list[str], optional
        A list of input column names for the UDF. If not provided, it will be
        inferred from the function signature. Or scan all columns.
    num_cpus: int, float, optional
        The (fraction) number of CPUs to acquire to run the job.
    num_gpus: int, float, optional
        The (fraction) number of GPUs to acquire to run the job.  Default 0.
    """
    if inspect.isclass(func):

        @functools.wraps(func)
        def _wrapper(*args, **kwargs) -> Union[UDF, functools.partial]:
            callable_obj = func(*args, **kwargs)
            return udf(
                callable_obj,
                cuda=cuda,
                data_type=data_type,
                version=version,
                field_metadata=field_metadata,
                input_columns=input_columns,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )

        return _wrapper  # type: ignore

    if func is None:
        return functools.partial(
            udf,
            cuda=cuda,
            data_type=data_type,
            version=version,
            field_metadata=field_metadata,
            input_columns=input_columns,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            **kwargs,
        )

    # we depend on default behavior of attrs to infer the output schema
    def _include_if_not_none(name, value) -> Dict[str, Any]:
        if value is not None:
            return {name: value}
        return {}

    args = {
        "func": func,
        "cuda": cuda,
        **_include_if_not_none("data_type", data_type),
        **_include_if_not_none("version", version),
        **_include_if_not_none("field_metadata", field_metadata),
        **_include_if_not_none("input_columns", input_columns),
        **_include_if_not_none("num_cpus", num_cpus),
        **_include_if_not_none("num_gpus", num_gpus),
    }
    # can't use functools.update_wrapper because attrs makes certain assumptions
    # and attributes read-only. We will figure out docs and stuff later
    return UDF(**args)


def _get_annotations(func: Callable) -> Dict[str, type]:
    if inspect.isfunction(func):
        return func.__annotations__
    elif isinstance(func, Callable):
        return func.__call__.__annotations__
    raise ValueError(f"func must be a function or a callable, got {func}")


def _is_batched_func(func: Callable) -> bool:
    annotations = _get_annotations(func)
    if "return" not in annotations:
        return False

    ret_type = annotations["return"]
    if ret_type != pa.Array and not isinstance(ret_type, pa.DataType):
        return False

    input_keys = list(annotations.keys() - {"return"})
    if len(input_keys) == 1:
        return all(
            annotations[input_key] in [pa.RecordBatch, pa.Array]
            for input_key in input_keys
        )

    if any(annotations[input_key] == pa.RecordBatch for input_key in input_keys):
        raise ValueError(
            "UDF can not have multiple parameters with 'pa.RecordBatch' type"
        )
    return all(annotations[input_key] in [pa.Array] for input_key in input_keys)


# Build numpy type mapping - numpy.bool deprecated in 1.x, reintroduced in 2.x
_NUMPY_TYPE_MAP = {
    bool: pa.bool_(),
    bytes: pa.binary(),
    float: pa.float32(),
    int: pa.int64(),
    str: pa.string(),
    numpy.bool_: pa.bool_(),
    numpy.uint8: pa.uint8(),
    numpy.uint16: pa.uint16(),
    numpy.uint32: pa.uint32(),
    numpy.uint64: pa.uint64(),
    numpy.int8: pa.int8(),
    numpy.int16: pa.int16(),
    numpy.int32: pa.int32(),
    numpy.int64: pa.int64(),
    numpy.float16: pa.float16(),
    numpy.float32: pa.float32(),
    numpy.float64: pa.float64(),
    numpy.str_: pa.string(),
}

# Add numpy.bool if available (numpy 2.x)
# In numpy 2.x, numpy.bool is a proper type, not a deprecated alias
# We only want to add it if numpy version is >= 2.0 to avoid deprecation warnings in 1.x
try:
    if int(numpy.__version__.split('.')[0]) >= 2:
        if hasattr(numpy, "bool"):
             _NUMPY_TYPE_MAP[numpy.bool] = pa.bool_()
except Exception:
    pass


def _infer_func_arrow_type(func: Callable, input_schema: pa.Schema) -> pa.DataType:
    """Infer the output schema of a UDF

    currently independent of the input schema, in the future we may want to
    infer the output schema based on the input schema, or the UDF itself could
    request the input schema to be passed in.
    """
    if isinstance(func, UDF):
        return func.data_type

    annotations = _get_annotations(func)
    if "return" not in annotations:
        raise ValueError(f"UDF {func} does not have a return type annotation")

    data_type = annotations["return"]
    # do dispatch to handle different types of output types
    # e.g. pydantic -> pyarrow type inference
    if isinstance(data_type, pa.DataType):
        return data_type

    if t := _NUMPY_TYPE_MAP.get(data_type):
        return t

    raise ValueError(f"UDF {func} has an invalid return type annotation {data_type}")

