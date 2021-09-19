import onnx
import argparse
import torch
from models.vad_models import LSTMModel


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='data/vad.pt', help='Path to saved checkpoint')
    namespace = parser.parse_args()
    argv = vars(namespace)

    torch_checkpoint = argv['checkpoint']
    model = LSTMModel().float()

    model.load_state_dict(torch.load(torch_checkpoint))
    model.eval()
    model.cpu()

    batch_size = 1
    x = torch.randn(batch_size, 394, 40, requires_grad=False)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "vad.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size', 1: 'audio_length'},  # variable length axes
                                    'output': {0: 'batch_size', 1: 'audio_length'}})

    onnx_model = onnx.load("vad.onnx")
    onnx.checker.check_model(onnx_model)
