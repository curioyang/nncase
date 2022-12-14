# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""nncase."""

from __future__ import annotations

import io
import re
import subprocess
import shutil
import os
import sys
from pathlib import Path
from shutil import which
from typing import List
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import _nncase
from _nncase import RuntimeTensor, TensorDesc, Simulator


def _check_env():#
    env = os.environ
    errors = []
    if not "NNCASE_COMPILER" in env:
        errors.append("NNCASE_COMPILER not found")
    return errors


def _initialize():
    errors = _check_env()
    if len(errors) > 0:
        raise Exception("check failed:\n" + str.join('\n', errors))
    _nncase.initialize(os.getenv("NNCASE_COMPILER"))


_initialize()
#_nncase.launch_debugger()


class ImportOptions:
    def __init__(self) -> None:
        pass


class PTQTensorOptions:
    calibrate_method: str
    input_mean: float
    input_std: float
    samples_count: int
    cali_data: list

    def __init__(self) -> None:
        pass

    def set_tensor_data(self, data: np.array) -> None:
        self.cali_data = [RuntimeTensor(d) for d in data]


class GraphEvaluator:
    _inputs: List[RuntimeTensor]
    _func: _nncase.Function
    _params: _nncase.Var
    _outputs: List[RuntimeTensor]

    def __init__(self, func: _nncase.Function) -> None:
        self._func = func
        self._params = func.parameters
        self._inputs = list([None] * len(self._params))
        self._outputs = None

    def get_input_tensor(self, index: int):
        assert index < len(self._inputs)
        tensor = self._inputs[index]
        return tensor.to_runtime_tensor() if tensor else None

    def set_input_tensor(self, index: int, value: RuntimeTensor):
        assert index < len(self._inputs)
        self._inputs[index] = _nncase.RTValue.from_runtime_tensor(value)

    def get_output_tensor(self, index: int):
        return self._outputs[index]

    def run(self):
        self._outputs = self._func.body.evaluate(self._inputs, self._params).to_runtime_tensors()

    @ property
    def outputs_size(self) -> int:
        return len(self._outputs)


class IRModule():
    _module: _nncase.IRModule = None

    def __init__(self, module: _nncase.IRModule):
        assert module.entry != None
        self._module = module

    @ property
    def entry(self) -> _nncase.IR.Function:
        return self._module.entry

    @ property
    def params(self) -> List[_nncase.IR.Var]:
        return self._module.parameters


class Compiler:
    _compiler: _nncase.Compiler
    _compile_options: _nncase.CompileOptions
    _module: IRModule

    def __init__(self) -> None:
        self._compile_options = _nncase.CompileOptions()
        self._compiler = _nncase.Compiler(self._compile_options)

    def set_compile_options(self, compile_options: CompileOptions):
        self.__process_compile_options(compile_options)

    def compile(self) -> None:
        self._compiler.compile()

    def create_evaluator(self, stage: int) -> GraphEvaluator:
        return GraphEvaluator(self._module.entry)

    def gencode(self, stream: io.RawIOBase) -> None:
        self._compiler.gencode(stream)

    def gencode_tobytes(self) -> bytes:
        code = io.BytesIO()
        self.gencode(code)
        return code.getvalue()

    def import_caffe(self, model: bytes, prototxt: bytes) -> None:
        raise NotImplementedError("import_caffe")

    def import_onnx(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.input_format = "onnx"
        self._import_module(model_content)

    def import_tflite(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.input_format = "tflite"
        self._import_module(model_content)

    def use_ptq(self, ptq_dataset_options: PTQTensorOptions, params: list) -> None:
        dataset = [data.to_nncase_tensor() for data in ptq_dataset_options.cali_data]
        dataset = _nncase.Compiler.PythonHelper.MakeDatasetProvider(
            dataset, ptq_dataset_options.samples_count, params)
        self.quant_options = _nncase.Compiler.PythonHelper.MakeQuantizeOptions(dataset)
        self._compiler.UsePTQ(self.quant_options)

    def dump_range_options(self) -> DumpRangeTensorOptions:
        raise NotImplementedError("dump_range_options")

    def __process_compile_options(self, compile_options: CompileOptions) -> ClCompileOptions:
        self._compile_options.target = compile_options.target
        self._compile_options.dump_level = 3 if compile_options.dump_ir == True else 0
        self._compile_options.dump_dir = compile_options.dump_dir

    def _import_module(self, model_content: bytes | io.RawIOBase) -> None:
        stream = io.BytesIO(model_content) if isinstance(model_content, bytes) else model_content
        self._module = IRModule(self._compiler.import_module(stream))


def check_target(target: str):
    def test_target(target: str):
        return target in ["cpu", "k510", "k230"]

    def target_exists(target: str):
        return _nncase.target_exists(target)

    return test_target(target) and target_exists(target)


class DumpRangeTensorOptions:
    calibrate_method: str
    samples_count: int

    def set_tensor_data(self, data: bytes):
        pass


class CalibMethod:
  NoClip: int = 0
  Kld: int = 1
  Random: int = 2

class ModelQuantMode:
  NoQuant: int = 0
  UsePTQ: int = 1
  UseQAT: int = 2

class ClQuantizeOptions():
  CalibrationDataset: object
  CalibrationMethod: CalibMethod
  BindQuantMethod: bool 
  UseSquant: bool 
  UseAdaRound  : bool 

class ClCompileOptions():
    InputFile: str
    InputFormat: str
    Target: str
    DumpLevel: int
    DumpDir: str
    QuantType: int
    WQuantType: int
    OutputFile: str
    ModelQuantMode: int
    QuantizeOptions: ClQuantizeOptions


class CompileOptions:
    benchmark_only: bool
    dump_asm: bool
    dump_dir: str
    dump_ir: bool
    swapRB: bool
    input_range: List[float]
    input_shape: List[int]
    input_type: str
    is_fpga: bool
    mean: List[float]
    std: List[float]
    output_type: str
    preprocess: bool
    quant_type: str
    target: str
    w_quant_type: str
    use_mse_quant_w: bool
    input_layout: str
    output_layout: str
    letterbox_value: float
    tcu_num: int

    def __init__(self) -> None:
        pass
