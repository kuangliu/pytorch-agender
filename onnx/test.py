import onnx

graph = onnx.load("./model/agnet.proto")
onnx.checker.check_graph(graph)
onnx.helper.printable_graph(graph)

# ...continuing from above
import onnx_caffe2.backend as backend
import numpy as np

rep = backend.prepare(graph, device="CPU") # or "CPU"
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class onnx_caffe2.backend.Workspace)
outputs = rep.run(np.ones((2, 3, 224, 224)).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print(outputs)
