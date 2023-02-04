import argparse
import onnx
import onnxruntime

def main(args):

    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)

    output =[node.name for node in onnx_model.graph.output]
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    print('Inputs: ', net_feed_input)
    print('Outputs: ', output, "\n")
    print("########################\n")

    ort_session_landmark = onnxruntime.InferenceSession(args.model_path)

    input_name = ort_session_landmark.get_inputs()[0].name
    print("input name", input_name)
    input_shape = ort_session_landmark.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = ort_session_landmark.get_inputs()[0].type
    print("input type", input_type, "\n")

    for i in range(len(output)):
     output_name = ort_session_landmark.get_outputs()[i].name
     print("output name:", output_name)
     output_shape = ort_session_landmark.get_outputs()[i].shape
     print("output shape:", output_shape)
     output_type = ort_session_landmark.get_outputs()[i].type
     print("output type:", output_type, "\n")
 

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('-m','--model_path',
                        default="./MASK_s.onnx",  #  checkpoint.pth.tar
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

