import bentoml
import numpy as np
import torch
from bentoml.io import JSON

runner = bentoml.pytorch.load_runner(
    "simple_gnn:latest",
    predict_fn_name="predict"
)

svc = bentoml.Service("simple_gnn", runners=[runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_arr: JSON):
    J_msg, b, msg_node, prob_gt = np.array(input_arr['J_msg']), np.array(input_arr['b']), np.array(input_arr['msg_node']), np.array(input_arr['prob_gt'])
    res = runner.run(J_msg, b, msg_node, prob_gt)
    return res

