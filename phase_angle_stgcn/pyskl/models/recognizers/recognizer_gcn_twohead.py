# Custom recognizer for the swimming phase-aware error classification

from .. import builder
from ..builder import RECOGNIZERS
from .base import BaseRecognizer
import torch
import torch.nn as nn
from .phase_timing_prior import apply_timing_prior


@RECOGNIZERS.register_module()
class RecognizerGCNTwoHead(BaseRecognizer):
    """GCN-based recognizer with two independent heads (soft coupling).

    Args:
        backbone (dict): Backbone config, e.g. dict(type='STGCN', ...).
        error_head (dict): Head config for the primary technique-error
            classification task, e.g. dict(type='GCNHead', num_classes=4,
            in_channels=256).
        phase_head (dict): Head config for the auxiliary phase
            classification task, e.g. dict(type='GCNHead', num_classes=4,
            in_channels=256).
        phase_loss_weight (float): Weight applied to the phase head's loss
            before summing with the error head's loss. Default: 0.3.
            This is a hyperparameter to tune -- start here and adjust based
            on validation performance; too high and the auxiliary task can
            dominate training, too low and it stops helping.
        train_cfg (dict): Training config. Default: dict().
        test_cfg (dict): Testing config. Default: dict().
    """

    def __init__(self,
                 backbone,
                 error_head,
                 phase_head,
                 phase_loss_weight=0.3,
                 train_cfg=dict(),
                 test_cfg=dict()):
        # cls_head=None: we deliberately don't use the single-head slot
        # BaseRecognizer provides. BaseRecognizer.__init__ also calls
        # self.init_weights() internally as its last step, which at that
        # point only initializes the backbone (error_head/phase_head don't
        # exist yet) -- see the explicit init calls below, which handle
        # both heads once they're built.
        super().__init__(backbone, cls_head=None, train_cfg=train_cfg, test_cfg=test_cfg)

        self.error_head = builder.build_head(error_head)
        self.phase_head = builder.build_head(phase_head)
        self.phase_loss_weight = phase_loss_weight

        self.error_head.init_weights()
        self.phase_head.init_weights()

    def init_weights(self):
        """Initialize backbone + both heads. Safe to call even before the
        heads exist yet (guarded by hasattr), since BaseRecognizer.__init__
        calls this once before our __init__ has built them."""
        self.backbone.init_weights()
        if hasattr(self, 'error_head'):
            self.error_head.init_weights()
        if hasattr(self, 'phase_head'):
            self.phase_head.init_weights()

    def extract_feat(self, keypoint):
        """Extract shared features through the backbone."""
        return self.backbone(keypoint)

    def forward_train(self, keypoint, label, phase_label=None, cycle_r=None, **kwargs):
        if phase_label is None:
            raise ValueError(
                "phase_label is required to train RecognizerGCNTwoHead. "
                "Check that: (1) your annotation pkl has a 'phase_label' "
                "field per sample, and (2) your pipeline's Collect(keys=...) "
                "and ToTensor(keys=...) both include 'phase_label'."
            )
        if cycle_r is None:
            raise ValueError(
                "cycle_r is required to train RecognizerGCNTwoHead. "
                "Check that: (1) your annotation pkl has a 'cycle_r' "
                "field per sample, and (2) your pipeline's Collect(keys=...) "
                "and ToTensor(keys=...) both include 'cycle_r'."
            )

        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        losses = dict()
        x = self.extract_feat(keypoint)  # raw backbone map (N, M, C, T, V)
        pooled = self._pool_backbone_features(x)

        #phase head predicts phase from raw backbone features
        phase_score = self.phase_head(pooled)
        phase_gt = phase_label.squeeze(-1)
        phase_losses = self.phase_head.loss(phase_score, phase_gt)
        for key, value in phase_losses.items():
            new_key = f'phase_{key}'
            if 'loss' in key:
                losses[new_key] = value * self.phase_loss_weight
            else:
                losses[new_key] = value

        #error head fused input = pooled backbone features + phase probs,
        #with ambiguous phase predictions corrected using the cycle-position
        #timing prior before being passed to the error head
        phase_probs = torch.softmax(phase_score, dim=-1)      # (N, 4)
        phase_probs, _, _, _ = apply_timing_prior(phase_probs, cycle_r.squeeze(-1))
        fused = torch.cat([pooled, phase_probs], dim=-1)      # (N, C+4)

        error_score = self.error_head(fused)  # already 2D, GCNHead skips its own pooling
        error_gt = label.squeeze(-1)
        error_losses = self.error_head.loss(error_score, error_gt)
        losses.update(error_losses)

        return losses

    def forward_test(self, keypoint, cycle_r=None, **kwargs):
        if cycle_r is None:
            raise ValueError(
                "cycle_r is required to test RecognizerGCNTwoHead. "
                "Check that: (1) your annotation pkl has a 'cycle_r' "
                "field per sample, and (2) your pipeline's Collect(keys=...) "
                "and ToTensor(keys=...) both include 'cycle_r'."
            )

        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        pooled = self._pool_backbone_features(x)
        phase_score = self.phase_head(pooled)
        phase_probs = torch.softmax(phase_score, dim=-1)

        # cycle_r is one fixed value per ORIGINAL sample (not per clip) --
        # UniformSample(num_clips=nc) produces nc different frame-samplings
        # of the same segment, but they all share the same cycle position,
        # so cycle_r arrives as shape (bs,) / (bs,1), never (bs, nc, ...).
        # Repeat each sample's value nc times to match the flattened
        # (bs*nc,) keypoint batch.
        cycle_r_flat = cycle_r.reshape(bs).unsqueeze(1).expand(bs, nc).reshape(bs * nc)
        phase_probs, _, _, _ = apply_timing_prior(phase_probs, cycle_r_flat)

        fused = torch.cat([pooled, phase_probs], dim=-1)

        error_score = self.error_head(fused)
        error_score = error_score.reshape(bs, nc, error_score.shape[-1])

        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        error_score = self.average_clip(error_score)

        return error_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, phase_label=None, cycle_r=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, phase_label=phase_label, cycle_r=cycle_r, **kwargs)

        return self.forward_test(keypoint, cycle_r=cycle_r, **kwargs)

    def _pool_backbone_features(self, x):
        """Replicates GCNHead's internal GCN-mode pooling: (N, M, C, T, V)
        -> (N, C). Needed because the error head no longer receives the raw
        backbone map directly -- it receives pooled features concatenated
        with the phase head's output, so pooling has to happen here instead
        of inside the head."""
        N, M, C, T, V = x.shape
        pool = nn.AdaptiveAvgPool2d(1)
        x = x.reshape(N * M, C, T, V)
        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        return x
