# @inproceedings{HSM-TDF,  
#   title={Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HSM-TDF}  
# }  

"""
Because the paper by Chen et al. [1] does not provide an official implementation, 
this file documents our reimplementation of the DeepFLD network architecture 
based on the methodological details reported therein.

[1] Chen, Y., Chen, X., Han, Y. et al. Multimodal Learning-based Prediction for Nonalcoholic Fatty Liver Disease. Mach. Intell. Res. 22, 871-887 (2025). https://doi.org/10.1007/s11633-024-1506-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class ConvBlock(nn.Module):
    """
    Two 3×3 conv layers with ReLU (no BN per paper), used inside each stage.
    After this block, a downsampling op (MaxPool2d, stride=2) halves H and W.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Stage(nn.Module):
    """One stage = ConvBlock (channels doubled) + 2× downsample by MaxPool2d(2).
    Matches: "each layer contains two 3×3 convs and a downsampling operation;
    after each layer, spatial dims are halved while channels are doubled".
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return x


class FeatureStandardizer(nn.Module):
    """
    Optional feature standardization layer.
    Paper states: "We first normalize all the input elements (subtract mean and divide variance)".
    - If mean/std are not provided, this is identity.
    - You can call set_stats(mean, std) later to register dataset-wide mean/std.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([], dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.tensor([], dtype=torch.float32), persistent=False)

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        assert mean.ndim == 1 and std.ndim == 1, "mean/std must be 1-D tensors"
        self.mean = mean.detach().float().clone()
        self.std = std.detach().float().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.numel() == 0 or self.std.numel() == 0:
            return x
        eps = 1e-6
        return (x - self.mean) / (self.std + eps)


class DeepFLD(nn.Module):
    """
    DeepFLD: Multimodal fusion network for NAFLD prediction (main task) with an auxiliary task.

    Architecture follows the paper's Section 3.3:
    - FEM (Feature Extraction Module):
        Conv1: 7×7 conv taking a 3-channel face image → 128 channels (no downsample here)
        4 stages; each stage = two 3×3 convs (channels doubled) + 2× downsample (MaxPool2d(2))
        After 4 stages: H,W halved 4 times (e.g., 512→256→128→64→32), channels doubled 4× (128→256→512→1024→2048)
        Global average pooling → 2048-D face feature, then Dropout(p=0.4)
    - Metadata embedding coder: Linear(metadata_dim → 8), producing an 8-D metadata feature
    - Fusion: Z_fusion = Concat(Z_image(2048), Z_metadata(8)) → 2056-D, optional standardization
    - FLM (Fatty Liver Module): two MLPs with ReLU + Dropout after each FC
        * FLMMLP1 (main task, NAFLD): 6 FC layers with hidden widths [2056, 1024, 1024, 512, 256, 128],
          followed by a 1-unit sigmoid output (fatty liver probability)
        * FLMMLP2 (auxiliary, image feature extraction only): input 2048,
          3 FC layers with hidden widths [2048, 1024, 1024]. Three heads predict:
          - gender (sigmoid, 1-dim)
          - BMI (linear, 1-dim)
          - weight (linear, 1-dim)

    Forward inputs:
      images: (B, 3, H, W) face images, ideally 512×512 for exact (32,32,2048) before pooling
      metadata: (B, metadata_dim) numeric indicator vector

    Forward outputs:
      main_out: (B,) float in [0,1], NAFLD probability (sigmoid)
      aux_out: dict with keys {'gender', 'bmi', 'weight'} mapping to (B,) tensors

    Notes:
      - Dropout p matches paper's stated 0.4.
      - No BatchNorm in convs to keep structure faithful to description.
      - If you want logits for BCEWithLogitsLoss, you can modify return_prob in forward.
    """

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 2,
        metadata_embed_dim: int = 512,   # chosen to match 2048 + 512
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        self.metadata_dim = metadata_dim
        self.metadata_embed_dim = metadata_embed_dim
        self.dropout_p = dropout_p

        # ===== FEM: image branch =====
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = Stage(128, 256)
        self.stage2 = Stage(256, 512)
        self.stage3 = Stage(512, 1024)
        self.stage4 = Stage(1024, 2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fem = nn.Dropout(p=self.dropout_p)

        # ===== Metadata embedding coder =====
        self.metadata_embed = nn.Linear(self.metadata_dim, self.metadata_embed_dim)

        # ===== Normalization (optional; dataset-level mean/std can be injected) =====
        self.fusion_norm = FeatureStandardizer()

        # ===== FLM: MLPs =====
        # Main task: FLMMLP1 hidden widths [2056, 1024, 1024, 512, 256, 128]
        self.mlp1_layers = nn.ModuleList()
        mlp1_in = 2048 + self.metadata_embed_dim
        mlp1_widths = [1024, 1024, 512, 256, 128]
        for w in mlp1_widths:
            self.mlp1_layers.append(nn.Linear(mlp1_in, w))
            mlp1_in = w
        self.mlp1_dropout = nn.Dropout(p=self.dropout_p)
        self.fc_main_out = nn.Linear(mlp1_widths[-1], num_classes)

        # Auxiliary task: FLMMLP2 hidden widths [2048, 1024, 1024], input uses ONLY image features (2048)
        self.mlp2_layers = nn.ModuleList()
        mlp2_widths = [1024, 1024]
        mlp2_in = 2048
        for w in mlp2_widths:
            self.mlp2_layers.append(nn.Linear(mlp2_in, w))
            mlp2_in = w
        self.mlp2_dropout = nn.Dropout(p=self.dropout_p)
        # Three output heads for auxiliary predictions
        self.fc_gender = nn.Linear(mlp2_widths[-1], 1)
        self.fc_bmi = nn.Linear(mlp2_widths[-1], 1)
        self.fc_weight = nn.Linear(mlp2_widths[-1], 1)

    # ----------------------------
    # Public helpers
    # ----------------------------
    def set_fusion_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Register dataset-level mean/std for Z_fusion standardization."""
        self.fusion_norm.set_stats(mean, std)

    # ----------------------------
    # Core forward
    # ----------------------------
    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        *,
        return_prob: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            images: (B, 3, H, W)
            metadata: (B, metadata_dim)
            return_prob: if True, main and gender outputs are sigmoid probabilities; else logits
        Returns:
            main_out, aux_out
              - main_out: (B,) NAFLD probability (or logits if return_prob=False)
              - aux_out: dict with keys 'gender' (B,), 'bmi' (B,), 'weight' (B,)
        """
        B = images.shape[0]

        # ---- FEM: image feature extraction ----
        x = self.relu(self.conv1(images))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)  # (B, 2048, 1, 1)
        z_img = x.view(B, -1)  # (B, 2048)
        z_img = self.dropout_fem(z_img)
        assert z_img.shape[1] == 2048, f"z_img dim expected 2048, got {z_img.shape[1]}"

        # ---- Metadata embedding ----
        assert metadata.dim() == 2 and metadata.shape[0] == B, "metadata must be (B, metadata_dim)"
        z_meta = self.metadata_embed(metadata)  # (B, 512)
        assert z_meta.shape[1] == self.metadata_embed_dim

        # ---- Fusion & optional normalization ----
        z_fusion = torch.cat([z_img, z_meta], dim=1)  # (B, 2048+512=2056)
        z_fusion = self.fusion_norm(z_fusion)

        # ---- FLMMLP1: main task ----
        h = z_fusion
        for fc in self.mlp1_layers:
            h = self.mlp1_dropout(self.relu(fc(h)))
        main_logit = self.fc_main_out(h).squeeze(-1)  # (B,)
        main_out = torch.sigmoid(main_logit) if return_prob else main_logit

        # ---- FLMMLP2: auxiliary task (image features only) ----
        h2 = z_img
        for fc in self.mlp2_layers:
            h2 = self.mlp2_dropout(self.relu(fc(h2)))
        gender_logit = self.fc_gender(h2).squeeze(-1)
        bmi = self.fc_bmi(h2).squeeze(-1)
        weight = self.fc_weight(h2).squeeze(-1)
        gender = torch.sigmoid(gender_logit)

        return main_out, gender, bmi, weight


if __name__ == "__main__":
    # Quick shape check
    model = DeepFLD(metadata_dim=23)  # e.g., 23 indicators used in the paper after selection
    imgs = torch.randn(2, 3, 512, 512)
    metas = torch.randn(2, 23)
    y_main, gender, bmi, weight = model(imgs, metas)
    print("main:", y_main.shape)
    print("aux gender/bmi/weight:", gender.shape, bmi.shape, weight.shape)
