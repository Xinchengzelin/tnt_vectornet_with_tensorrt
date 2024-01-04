'''
Author: zhanghao
LastEditTime: 2023-04-18 14:30:40
FilePath: /my_vectornet_github/model/layers/subgraph.py
LastEditors: zhanghao
Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mlp import MLP
from typing import Optional, Tuple
from torch_scatter import scatter_max ,scatter
import torch_scatter
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


# @parse_args('v','v','i','none','none','s')
# def scatter(g,
#             src,
#             index,
#             dim,
#             out,
#             dim_size,
#             reduce):
#     # Define how the custom operation should be mapped to ONNX
#     return g.op('customOp::scatter_max', src, index=index,dim=dim,reduce)
# register_custom_op_symbolic("torch_scatter::scatter_max", scatter, 12)

class CustomOpImpl(torch.autograd.Function):
    @staticmethod 
    def symbolic(g: torch.Graph, src: torch.Tensor, index: torch.Tensor, dim:int) -> torch.Tensor: # 这里注册了，导出onnx， autograd不会跟踪了
        return g.op("plugin::ScatterMax", src, index, dim_i=dim, out=None, dim_size=None) #

    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor, dim: int) ->  Tuple[torch.Tensor, torch.Tensor]:
        # ctx.save_for_backward(src)
        return torch.ops.torch_scatter.scatter_max(src, index, dim, None, None)[0]
    
class CustomOp(nn.Module):
    def __init__(self,  dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x, index):
        return CustomOpImpl.apply(x, index, self.dim)


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, x, cluster):
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                # print(f"x dtype: {type(x)} cluster dtype: {type(cluster)}")
                # x_max = scatter(x, cluster, dim=0, reduce='max')
                # x_max = scatter_max(x, index=cluster)[0]
                x_max = CustomOp(dim=0)(x,cluster)
                # print("1 x_max shape: ", x_max.shape)
                x = torch.cat([x, x_max[cluster]], dim=-1)

        # print("subgraph before linear: \n")
        # for xx in x:
        #     print(xx[:64])
        # print("\n\n\n")

        x = self.linear(x)
        # x = scatter(x, cluster, dim=0, reduce='max')
        # x = scatter_max(x, index=cluster)[0]
        x = CustomOp(dim=0)(x,cluster)
        print(f"x dtype: {type(x)} cluster dtype: {type(cluster)}")
        # print("2 x shape: ", x.shape)

        
        # print("subgraph after norm: \n", F.normalize(x, p=2.0, dim=1))
        
        return F.normalize(x, p=2.0, dim=1)  # L2 normalization


if __name__ == "__main__":
    layer = SubGraph(in_channels=6, num_subgraph_layers=1, hidden_unit=64)
    data = torch.randn((11, 6))
    cluster = torch.cat((torch.zeros(6), torch.ones(5))).long()
    out = layer(data, cluster)
    print(out.shape)

    EXPORT = 1
    if EXPORT:
        import onnx
        from onnxsim import simplify

        layer.eval()
        torch.onnx._export(
            layer,
            (data, cluster),
            "t.onnx",
            input_names=["data", ],
            output_names=["output"],
            dynamic_axes=None,
            opset_version=11,
        )
        print("export done.")

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load("t.onnx")
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "t.onnx")
        print("simplify done.")
