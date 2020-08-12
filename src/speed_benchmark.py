""" Speed benchmark for pure pytorch and trt optimized model. """
import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import numpy as np
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch

from model import UnetResnet34

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
NUM_SAMPLES = 500
INPUT_SHAPE = (1, 3, 320, 320)


def allocate_buffers(engine):
    host_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), trt.nptype(engine.get_binding_dtype(0)))
    host_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                        trt.nptype(engine.get_binding_dtype(0)))

    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)

    stream = cuda.Stream()

    return host_input, device_input, host_output, device_output, stream


def trt_benchmark(engine_path):
    engine = None
    with open(engine_path, 'rb') as f:
        with trt.Runtime(TRT_LOGGER) as trt_runtime:
            engine = trt_runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("Error after engine deserializing, stopping benchmark...")
        exit(-1)

    timings = []
    with engine.create_execution_context() as context:
        for _ in range(NUM_SAMPLES):
            data = np.random.random(INPUT_SHAPE).astype(np.float32).reshape(-1)
            host_input, device_input, host_output, device_output, stream = allocate_buffers(engine)

            cuda.memcpy_htod(device_input, data)

            start = time.time()
            context.execute(1, [int(device_input), int(device_output)])
            timings.append(time.time() - start)

            cuda.memcpy_dtoh(host_output, device_output)

    print("TRT inference mean time: {:.5f} sec, std: {:.5f}".format(np.mean(timings), np.std(timings)))


def pytorch_benchmark(path_to_model):
    unet = UnetResnet34()
    unet.load_state_dict(torch.load(path_to_model))

    device = torch.device("cuda:0")
    unet.to(device)

    unet.eval()

    timings = []
    for _ in range(NUM_SAMPLES):
        t = torch.randn(INPUT_SHAPE, device=device)
        start = time.time()
        unet(t)
        timings.append(time.time() - start)

    print("Pytorch inference mean time: {:.5f} sec, std: {:.5f}".format(np.mean(timings), np.std(timings)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-model', help='Path to pytorch model', required=True, type=str)
    parser.add_argument('--trt-engine', help='Path to trt model', required=True, type=str)

    args = parser.parse_args()

    print("Starting pytorch speed benchmark...")
    pytorch_benchmark(args.torch_model)

    print("Starting trt speed benchmark...")
    trt_benchmark(args.trt_engine)
