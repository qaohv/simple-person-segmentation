""" Converts model from pytorch to trt """
import argparse
from distutils.util import strtobool
from pathlib import Path

import onnx
import onnxsim
import tensorrt as trt
import torch
from model import UnetResnet34

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model', help='Path to pytorch model', required=True, type=str)
    parser.add_argument('--output-model', help='Path to trt model', required=True, type=str)
    parser.add_argument('--check-onnx', help='Checks onnx file correctness (defualt true)', type=strtobool,
                        default=True)
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for trt model (default: 1)')

    args = parser.parse_args()

    device = torch.device("cuda:0")

    model = UnetResnet34()
    model.load_state_dict(torch.load(args.input_model))
    model.eval()

    model = model.to(device)
    print("Exporting model to onnx...")

    input_model_path = Path(args.input_model)
    onnx_model_path = input_model_path.parent / (input_model_path.stem + '.onnx')

    t = torch.randn((1, 3, 320, 320), device=device)
    torch.onnx.export(model, t, str(onnx_model_path), export_params=True, opset_version=10)

    model_opt, checks_passed = onnxsim.simplify(str(onnx_model_path), check_n=3)
    if not checks_passed:
        print("Checks failed")
        exit(-1)

    onnx.save(model_opt, onnx_model_path)

    print("Creating trt engine...")

    engine = None
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network(explicit_batch_flag) as network:
            with trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_batch_size = args.batch_size
                builder.max_workspace_size = 1 << 30

                with open(str(onnx_model_path), "rb") as onnx_file:
                    if not parser.parse(onnx_file.read()):
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))

                    engine = builder.build_cuda_engine(network)
                    print(engine.get_binding_shape(0), engine.get_binding_shape(1))

    if engine:
        print("Saving engine to file...")
        with open(args.output_model, 'wb') as f:
            f.write(engine.serialize())
    else:
        print("Engine is None, smth went wrong.")
