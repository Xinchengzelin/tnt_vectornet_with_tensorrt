import torch
import sys
import os
from typing import Optional, Tuple
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
print(sys.path)

from vectornet_export_wts import VectorNetExport,load_vectornet


if __name__ == "__main__":
    # model = VectorNetExport(in_channels=6, horizon=30)
    ckpt = "/media/zetlin/Data2/Code/vectornet/tnt_vectornet_with_tensorrt/weights/sg_best_vectornet.pth"
    model = load_vectornet(ckpt)

    x = torch.tensor([-0.3801, -0.1300, 1.1666,  -1.1327, 0.6438, 0.6729,  -1.1299, -2.2857, 0.1849,  0.0493,
                    -0.4179, -0.5331, 0.7467,  -1.0006, 1.4848, 0.2771,  0.1393,  -0.9162, -1.7744, 0.8850,
                    -1.6748, 1.3581,  -0.4987, -0.7244, 0.7941, -0.4109, -0.3446, -0.5246, -0.8153, -0.5685,
                    1.9105,  -0.1069, 0.7214,  0.5255,  0.3654, -0.3434, 0.7163,  -0.6460, 1.9680,  0.8964,
                    0.3845,  3.4347,  -2.6291, -0.9330, 0.6411, 0.9983,  0.6731,  0.9110,  -2.0634, -0.5751,
                    1.4070,  0.5285,  -0.1171, -0.1863, 2.1200, 1.3745,  0.9763,  -0.1193, -0.3343, -1.5933]).reshape(-1,6)
    # cluster = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3, 3, 4])\
    cluster = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3, 4, 5])
    # id_embedding = torch.randn((5,2))
    id_embedding = torch.arange(12).reshape(6,2)

    # print("id_embedding = \n", id_embedding)
    # print("x = \n", x.reshape(1,-1))

    out = model(x, cluster, id_embedding)
    print(out.shape, "\n", out)


    onnx_path = "vectornet.onnx"

    model.eval()
    torch.onnx.export(
        model,
        (x, cluster, id_embedding),
        onnx_path,
        input_names=["x, cluster, id_embedding"],
        output_names=["trajs"],
        dynamic_axes=None,
        opset_version=12,
    )
    print("export done!!!!")