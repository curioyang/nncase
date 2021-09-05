from typing import Dict, List, Tuple, Union, Any
from itertools import product
import re
import yaml
from pathlib import Path
import numpy as np
import os
import shutil
from abc import ABCMeta, abstractmethod
import nncase
import struct
from compare_util import compare
import copy
import tensorflow as tf


class Edict:
    def __init__(self, d: Dict[str, int]) -> None:
        assert(isinstance(d, dict)), "the Edict only accepct Dict for init"
        for name, value in d.items():
            if isinstance(value, (list, tuple)):
                setattr(self, name,
                        [Edict(x) if isinstance(x, dict) else x for x in value])
            else:
                if 'kwargs' in name:
                    setattr(self, name, value if value else dict())
                else:
                    if isinstance(value, dict):
                        setattr(self, name, Edict(value))
                    else:
                        setattr(self, name, value)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __repr__(self, indent=0) -> str:
        s: str = ''
        for k, v in self.__dict__.items():
            s += indent * ' ' + k + ' : '
            if isinstance(v, Edict):
                s += '\n' + v.__repr__(len(s) - s.rfind('\n'))
            else:
                s += v.__repr__().replace('\n', ' ')
            s += '\n'
        return s.rstrip('\n')

    @property
    def dict(self):
        return self.__dict__

    def update(self, d: Dict):
        for name, new_value in d.items():
            if name in self.keys():
                if isinstance(new_value, dict):
                    old_value = getattr(self, name)
                    if old_value is None:
                        setattr(self, name, Edict(new_value))
                    elif isinstance(old_value, (Edict, dict)):
                        old_value.update(new_value)
                elif isinstance(new_value, (list, tuple)) and name == 'specifics':
                    if getattr(self, name) == None:
                        setattr(self, name, [])
                    assert(hasattr(self, 'common')
                           ), "The specifics new_value need common dict to overload !"
                    common = getattr(self, 'common')
                    for specific in new_value:
                        import_common = copy.deepcopy(common)
                        import_common.update(specific)
                        getattr(self, name).append(import_common)
                else:
                    setattr(self, name, new_value)
            else:
                setattr(self, name, new_value)


def generate_random(shape: List[int], dtype: np.dtype,
                    abs: bool = False) -> np.ndarray:
    if dtype is np.uint8:
        data = np.random.randint(0, 256, shape)
    elif dtype is np.int8:
        data = np.random.randint(-128, 128, shape)
    else:
        data = np.random.rand(*shape)
    data = data.astype(dtype=dtype)
    if abs:
        return np.abs(data)
    return data


def _cast_bfloat16_then_float32(values: np.array):
    shape = values.shape
    values = values.reshape([-1])
    for i, value in enumerate(values):
        value = float(value)
        value = 1
        packed = struct.pack('!f', value)
        integers = [c for c in packed][:2] + [0, 0]
        value = struct.unpack('!f', bytes(integers))[0]
        values[i] = value

    values = values.reshape(shape)
    return values


DataFactory = {
    'generate_random': generate_random
}


class TestRunner(metaclass=ABCMeta):
    def __init__(self, case_name, targets=None, overwirte_configs: Union[Dict, str] = None) -> None:
        config_root = os.path.dirname(__file__)
        with open(os.path.join(config_root, 'config.yml'), encoding='utf8') as f:
            cfg: dict = yaml.safe_load(f)
            config = Edict(cfg)
        config = self.update_config(config, overwirte_configs)
        self.cfg = self.validte_config(config)

        case_name = case_name.replace('[', '_').replace(']', '_')
        self.case_dir = os.path.join(self.cfg.setup.root, case_name)
        self.clear(self.case_dir)

        self.validate_targets(targets)

        self.inputs: List[Dict] = []
        self.calibs: List[Dict] = []
        self.outputs: List[Dict] = []
        self.input_paths: List[Tuple[str, str]] = []
        self.calib_paths: List[Tuple[str, str]] = []
        self.output_paths: List[Tuple[str, str]] = []
        self.model_type: str = ""
        self.pre_process: List[Dict] = []

        self.num_pattern = re.compile("(\d+)")

    def transform_input(self, values: np.array, type: str, stage: str):
        values = copy.deepcopy(values)
        if(len(values.shape) == 4 and self.pre_process[0]['flag']):
            if stage == "CPU":
                # onnx \ caffe
                if self.model_type == "onnx" or self.model_type == "caffe" or (self.model_type == "tflite" and self.cfg.case.importer_opt.kwargs['input_layout'] == "NCHW"):
                    values = np.transpose(values, [0, 3, 1, 2])
            else:
                if self.cfg.case.importer_opt.kwargs['input_layout'] == "NCHW":
                    values = np.transpose(values, [0, 3, 1, 2])

            if type == 'float32':
                return values.astype(np.float32)
            elif type == 'uint8':
                values = ((values) * 255).astype(np.uint8)
                return values
            elif type == 'int8':
                values = (values * 255 - 128).astype(np.int8)
                return values
            else:
                raise TypeError(" Not support type for quant input")
        else:
            return values

    def get_process_config(self, config):
        # preprocess flag
        preprocess_flag = {}
        preprocess_flag['flag'] = config['flag']

        # dequant
        process_deq = {}
        process_deq['range'] = config['input_range']
        process_deq['input_type'] = config['input_type']

        # norm
        process_norm = {}
        data = {}
        data = {
            'mean': config['mean'],
            'scale': config['scale']
        }
        process_norm['norm'] = data

        # bgr2rgb
        process_format = {}
        process_format['image_format'] = config['image_format']

        # letter box
        process_letterbox = {}
        process_letterbox['input_range'] = config['input_range']
        process_letterbox['input_shape'] = config['input_shape']
        process_letterbox['shape'] = self.inputs[0]['shape']
        process_letterbox['input_type'] = config['input_type']

        self.pre_process.append(preprocess_flag)
        self.pre_process.append(process_deq)
        self.pre_process.append(process_format)
        self.pre_process.append(process_letterbox)
        self.pre_process.append(process_norm)

    def data_pre_process(self, data):
        data = copy.deepcopy(data)
        if self.pre_process[0]['flag'] and len(data.shape) == 4:
            if self.cfg.case.compile_opt.kwargs['input_type'] == "uint8":
                data *= 255.
            # elif self.cfg.case.compile_opt.kwargs['input_type'] == "int8":
            #     data *= 255.
            #     data -= 128.
            for item in self.pre_process:
                # dequantize
                if 'range' in item.keys() and 'input_type' in item.keys():
                    Q_max, Q_min = 0, 0
                    if item['input_type'] == 'uint8':
                        Q_max, Q_min = 255, 0
                    # elif item['input_type'] == 'int8':
                    #     Q_max, Q_min = 127, -128
                    else:
                        continue
                    scale = (item['range'][1] - item['range'][0]) / (Q_max - Q_min)
                    bias = round((item['range'][1] * Q_min - item['range'][0] *
                                  Q_max) / (item['range'][1] - item['range'][0]))
                    data *= scale
                    data -= bias

                # BGR2RGB
                if 'image_format' in item.keys():
                    if item['image_format'] == 'BGR' and data.shape[-1] == 3:
                        data = data[:, :, :, ::-1]
                        data = np.array(data)

                # LetterBox
                if 'input_range' in item.keys() and 'input_shape' in item.keys() and 'shape' in item.keys():
                    model_shape: List = []
                    if self.model_type == "onnx" or self.model_type == "caffe":
                        model_shape = [1, item['shape'][2], item['shape'][3], item['shape'][1]]
                    else:
                        model_shape = item['shape']
                    if model_shape[1] != data.shape[1] or model_shape[2] != data.shape[2]:
                        in_h, in_w = data.shape[1], data.shape[2]
                        model_h, model_w = model_shape[1], model_shape[2]
                        ratio = min(model_h / in_h, model_w / in_w)
                        resize_shape = data.shape[0], round(in_h * ratio), round(in_w * ratio), 3

                        resize_data = tf.image.resize(
                            data[0], [resize_shape[1], resize_shape[2]], method=tf.image.ResizeMethod.BILINEAR)
                        dh = model_shape[1] - resize_shape[1]
                        dw = model_shape[2] - resize_shape[2]
                        dh /= 2
                        dw /= 2

                        resize_data = np.array(resize_data, dtype=np.float32)

                        data = tf.image.pad_to_bounding_box(resize_data, round(
                            dh - 0.1), round(dw - 0.1), model_h, model_w)

                        data = np.array(data, dtype=np.float32)
                        data = np.expand_dims(data, 0)

                # Normalize(Standardization)
                if 'norm' in item.keys():
                    for i in range(data.shape[-1]):
                        k = i
                        if data.shape[-1] > 3:
                            k = 0
                        data[:, :, :, i] = (data[:, :, :, i] - float(item['norm']['mean'][k])) / \
                            float(item['norm']['scale'][k])

        return data

    def validte_config(self, config):
        in_ci = os.getenv('CI', False)
        if in_ci:
            config.judge.common.log_hist = False
            config.setup.log_txt = False
        return config

    def update_config(self, config: Edict, overwirte_configs: Dict) -> Edict:
        if overwirte_configs:
            if isinstance(overwirte_configs, str):
                overwirte_configs: dict = yaml.safe_load(overwirte_configs)
            config.update(overwirte_configs)
        return config

    def validate_targets(self, targets: List[str]):
        def _validate_targets(old_targets: List[str]):
            new_targets = []
            for t in old_targets:
                if nncase.test_target(t):
                    new_targets.append(t)
                else:
                    print("WARN: target[{0}] not found".format(t))
            return new_targets
        self.cfg.case.eval[0].values = _validate_targets(
            targets if targets else self.cfg.case.eval[0].values)
        self.cfg.case.infer[0].values = _validate_targets(
            targets if targets else self.cfg.case.infer[0].values)

    def run(self, model_path: Union[List[str], str]):
        # TODO add mulit process pool
        # case_name = self.process_model_path_name(model_path)
        # case_dir = os.path.join(self.cfg.setup.root, case_name)
        # if not os.path.exists(case_dir):
        #     os.makedirs(case_dir)
        if isinstance(model_path, str):
            case_dir = os.path.dirname(model_path)
        elif isinstance(model_path, list):
            case_dir = os.path.dirname(model_path[0])
        self.run_single(self.cfg.case, case_dir, model_path)

    def process_model_path_name(self, model_path: str) -> str:
        if Path(model_path).is_file():
            case_name = Path(model_path)
            return '_'.join(str(case_name.parent).split('/') + [case_name.stem])
        return model_path

    def clear(self, case_dir):
        in_ci = os.getenv('CI', False)
        if in_ci:
            if os.path.exists(self.cfg.setup.root):
                shutil.rmtree(self.cfg.setup.root)
        else:
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir)
        os.makedirs(case_dir)

    @abstractmethod
    def parse_model_input_output(self, model_path: Union[List[str], str]):
        pass

    @abstractmethod
    def cpu_infer(self, case_dir: str, model_content: Union[List[str], str]):
        pass

    @ abstractmethod
    def import_model(self, compiler, model_content, import_options):
        pass

    def run_single(self, cfg, case_dir: str, model_file: Union[List[str], str]):
        if not self.inputs:
            self.parse_model_input_output(model_file)
        names_im, args_im = TestRunner.split_value(cfg.importer_opt)
        names_pre, args_pre = TestRunner.split_value(cfg.preprocess_opt)
        names = names_im + names_pre
        args = args_im + args_pre
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            self.get_process_config(dict_args)
            if dict_args['flag'] and dict_args['input_shape'] != None:
                self.generate_data(cfg.generate_inputs, case_dir,
                                   self.inputs, self.input_paths, 'input', dict_args['input_shape'])
                self.generate_data(cfg.generate_calibs, case_dir,
                                   self.calibs, self.calib_paths, 'calib', dict_args['input_shape'])
            else:
                self.generate_data(cfg.generate_inputs, case_dir,
                                   self.inputs, self.input_paths, 'input')
                self.generate_data(cfg.generate_calibs, case_dir,
                                   self.calibs, self.calib_paths, 'calib')
            self.cpu_infer(case_dir, model_file, dict_args['input_type'])
            import_options, compile_options = self.get_compiler_options(dict_args, model_file)
            # print(import_options)
            model_content = self.read_model_file(model_file)
            self.run_evaluator(cfg, case_dir, import_options, compile_options, model_content)
            self.run_inference(cfg, case_dir, import_options,
                               compile_options, model_content, dict_args)

    def get_compiler_options(self, cfg, model_file):
        import_options = nncase.ImportOptions()
        if isinstance(model_file, str):
            if os.path.splitext(model_file)[-1] == ".tflite":
                import_options.input_layout = cfg['input_layout']
                import_options.output_layout = cfg['output_layout']
            elif os.path.splitext(model_file)[-1] == ".onnx":
                import_options.input_layout = cfg['input_layout']
                import_options.output_layout = cfg['output_layout']
        elif isinstance(model_file, list):
            if os.path.splitext(model_file[1])[-1] == ".caffemodel":
                import_options.input_layout = cfg['input_layout']
                import_options.output_layout = cfg['output_layout']

        compile_options = nncase.CompileOptions()
        for k, v in cfg.items():
            e = '"'
            if k not in ['input_layout', 'output_layout', 'flag']:
                exec(f"compile_options.{k} = {e + v + e if isinstance(v, str) else v}")
        return import_options, compile_options

    def run_evaluator(self, cfg, case_dir, import_options, compile_options, model_content):
        names, args = TestRunner.split_value(cfg.eval)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            eval_output_paths = self.generate_evaluates(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args)
            judge, result = self.compare_results(
                self.output_paths, eval_output_paths, dict_args)
            assert(judge), 'Fault result in eval' + result

    def run_inference(self, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        names, args = TestRunner.split_value(cfg.infer)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            if dict_args['ptq'] and len(self.inputs) != 1:
                continue

            infer_output_paths = self.nncase_infer(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args, preprocess_opt)
            judge, result = self.compare_results(
                self.output_paths, infer_output_paths, dict_args)
            assert(judge), 'Fault result in infer' + result

    @ staticmethod
    def split_value(kwcfg: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        arg_names = []
        arg_values = []
        for d in kwcfg:
            arg_names.append(d.name)
            arg_values.append(d.values)
        return (arg_names, arg_values)

    def read_model_file(self, model_file: Union[List[str], str]):
        if isinstance(model_file, str):
            with open(model_file, 'rb') as f:
                return f.read()
        elif isinstance(model_file, list):
            model_content = []
            with open(model_file[0], 'rb') as f:
                model_content.append(f.read())
            with open(model_file[1], 'rb') as f:
                model_content.append(f.read())
            return model_content

    @ staticmethod
    def kwargs_to_path(path: str, kwargs: Dict[str, str]):
        for k, v in kwargs.items():
            if isinstance(v, str):
                path = os.path.join(path, v)
            elif isinstance(v, bool):
                path = os.path.join(path, ('' if v else 'no') + k)
        return path

    def generate_evaluates(self, cfg, case_dir: str,
                           import_options: nncase.ImportOptions,
                           compile_options: nncase.CompileOptions,
                           model_content: Union[List[bytes], bytes], kwargs: Dict[str, str]
                           ) -> List[Tuple[str, str]]:
        eval_dir = TestRunner.kwargs_to_path(
            os.path.join(case_dir, 'eval'), kwargs)
        compile_options.target = kwargs['target']
        compile_options.dump_dir = eval_dir
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)
        evaluator = compiler.create_evaluator(3)
        eval_output_paths = []
        for i in range(len(self.inputs)):
            input_tensor = nncase.RuntimeTensor.from_numpy(
                self.transform_input(self.data_pre_process(self.inputs[i]['data']), "float32", "CPU"))
            # self.data_pre_process(self.inputs[i]['data']))
            input_tensor.copy_to(evaluator.get_input_tensor(i))
            evaluator.run()

        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            eval_output_paths.append((
                os.path.join(eval_dir, f'nncase_result_{i}.bin'),
                os.path.join(eval_dir, f'nncase_result_{i}.txt')))
            result.tofile(eval_output_paths[-1][0])
            self.totxtfile(eval_output_paths[-1][1], result)
        return eval_output_paths

    def nncase_infer(self, cfg, case_dir: str,
                     import_options: nncase.ImportOptions,
                     compile_options: nncase.CompileOptions,
                     model_content: Union[List[bytes], bytes],
                     kwargs: Dict[str, str],
                     preprocess: Dict[str, str]
                     ) -> List[Tuple[str, str]]:
        infer_dir = TestRunner.kwargs_to_path(
            os.path.join(case_dir, 'infer'), kwargs)
        compile_options.target = kwargs['target']
        compile_options.dump_dir = infer_dir
        compile_options.input_type = preprocess['input_type']
        compile_options.quant_type = cfg.compile_opt.quant_type
        compile_options.image_format = preprocess['image_format']
        compile_options.input_shape = preprocess['input_shape']
        compile_options.input_range = preprocess['input_range']
        compile_options.preprocess = preprocess['flag']
        compile_options.mean = preprocess['mean']
        compile_options.scale = preprocess['scale']
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.calibs]).tobytes())
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            ptq_options.input_mean = cfg.ptq_opt.kwargs['input_mean']
            ptq_options.input_std = cfg.ptq_opt.kwargs['input_std']

            compiler.use_ptq(ptq_options)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)
        sim = nncase.Simulator()
        sim.load_model(kmodel)
        infer_output_paths: List[np.ndarray] = []
        for i in range(len(self.inputs)):
            sim.set_input_tensor(
                i, nncase.RuntimeTensor.from_numpy(self.transform_input(self.inputs[i]['data'], preprocess['input_type'], "infer")))
        sim.run()

        for i in range(sim.outputs_size):
            result = sim.get_output_tensor(i).to_numpy()
            infer_output_paths.append((
                os.path.join(infer_dir, f'nncase_result_{i}.bin'),
                os.path.join(infer_dir, f'nncase_result_{i}.txt')))
            result.tofile(infer_output_paths[-1][0])
            self.totxtfile(infer_output_paths[-1][1], result)
        return infer_output_paths

    def on_test_start(self) -> None:
        pass

    def generate_data(self, cfg, case_dir: str, inputs: List[Dict], path_list: List[str], name: str, input_shape: List = []):
        for n in range(cfg.numbers):
            i = 0
            for input in inputs:
                shape = []
                if input_shape != [] and len(input_shape) == 3 and self.cfg.case.preprocess_opt.flag:
                    shape = [1, input_shape[0], input_shape[1], input_shape[2]]
                else:
                    shape = input['shape']
                shape[0] *= cfg.batch_size
                data = DataFactory[cfg.name](shape, input['dtype'], **cfg.kwargs)

                path_list.append(
                    (os.path.join(case_dir, f'{name}_{n}_{i}.bin'),
                     os.path.join(case_dir, f'{name}_{n}_{i}.txt')))
                data.tofile(path_list[-1][0])
                self.totxtfile(path_list[-1][1], data)
                i += 1
                input['data'] = data

    def process_input(self, inputs: List[np.array], **kwargs) -> None:
        pass

    def process_output(self, outputs: List[np.array], **kwargs) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def compare_results(self,
                        ref_ouputs: List[Tuple[str]],
                        test_outputs: List[Tuple[str]],
                        kwargs: Dict[str, str]) -> Tuple[bool, str]:

        judeg_cfg = copy.deepcopy(self.cfg.judge.common)
        if self.cfg.judge.specifics:
            for specific in self.cfg.judge.specifics:
                if kwargs['target'] in specific.matchs.dict['target'] and kwargs['ptq'] == specific.matchs.dict['ptq']:
                    judeg_cfg.update(specific)
                    break

        for ref_file, test_file in zip(ref_ouputs, test_outputs):

            judge, simarity_info = compare(test_file, ref_file,
                                           judeg_cfg.simarity_name,
                                           judeg_cfg.threshold,
                                           judeg_cfg.log_hist)
            name_list = test_file[1].split(os.path.sep)
            kw_names = ' '.join(name_list[-len(kwargs) - 2:-1])
            i = self.num_pattern.findall(name_list[-1])
            result_info = "\n{0} [ {1} ] Output: {2}!!\n".format(
                'Pass' if judge else 'Fail', kw_names, i)
            result = simarity_info + result_info
            # print(result) temp disable
            with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
                f.write(result)
            if not judge:
                return False, result
        return True, result

    def totxtfile(self, save_path, value_np: np.array, bit_16_represent=False):
        if self.cfg.setup.log_txt:
            if bit_16_represent:
                np.save(save_path, _cast_bfloat16_then_float32(value_np))
            else:
                np.savetxt(save_path, value_np.flatten(), fmt='%f', header=str(value_np.shape))
            print("----> %s" % save_path)
