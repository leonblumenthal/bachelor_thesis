from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
from yolact_edge.data.config import cfg, set_cfg
from yolact_edge.inference import parse_args
from yolact_edge.layers.output_utils import postprocess
from yolact_edge.utils.augmentations import BaseTransform, FastBaseTransform
from yolact_edge.utils.tensorrt import convert_to_tensorrt
from yolact_edge.yolact import Yolact


class YolactEdgePredictor:
    """This is a modified version of the YOLACTEdgeInference (quick and dirty)"""

    def __init__(self, model_config: str, weights_path: str):
        global cfg
        set_cfg(model_config)
        config_ovr = {'use_fast_nms': True, 'mask_proto_debug': False}
        cfg.replace(cfg.copy(config_ovr))

        global args
        args = parse_args('')

        with torch.no_grad():
            cudnn.fastest = True
            cudnn.deterministic = True
            cudnn.benchmark = False
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            net = Yolact(training=False)
            net.load_weights(weights_path, args=args)
            net.eval()
            convert_to_tensorrt(net, cfg, args, transform=BaseTransform())
            net = net.cuda()
            self.net = net

    def __call__(
        self, img
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run yolact edge and return classes, scores, boxes, and masks."""

        frame = torch.Tensor(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        extras = {
            "backbone": "full",
            "interrupt": False,
            "keep_statistics": False,
            "moving_statistics": None,
        }

        with torch.no_grad():
            preds = self.net(batch, extras=extras)["pred_outs"]
            h, w = frame.shape[:2]
            out = postprocess(preds, w, h)

        return out
