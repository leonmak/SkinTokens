"""Cog entrypoint for TokenRig (SkinTokens).

This runs the same inference path as `demo.py`: a `bpy` sidecar server owns all
Blender I/O (mesh import, rigged export, rig transfer) while the TokenRig
autoregressive model generates the skeleton and skin tokens on the GPU.

Run locally with:

    cog predict -i mesh=@examples/giraffe.glb
"""

import atexit
import os
import pathlib
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
# Checkpoint configs reference paths relative to the repo root (the Qwen3 config
# in models/, the FSQ-CVAE checkpoint in experiments/), so pin the working
# directory before anything imports or loads them.
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import requests
import torch
from cog import BasePredictor, Input, Path
from torch import Tensor

from download import MODELS as HF_MODEL_FILES, LLM_LOCAL_DIR, download_llm, download_model
from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.data.vertex_group import voxel_skin
from src.model.tokenrig import TokenRigResult
from src.server.spec import BPY_SERVER, bytes_to_object, get_model, object_to_bytes
from src.tokenizer.parse import get_tokenizer

MODEL_CKPT = os.environ.get(
    "SKINTOKENS_CKPT",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
)
SUPPORTED_EXT = {".obj", ".fbx", ".glb"}
BPY_STARTUP_TIMEOUT = 120


def ensure_weights() -> None:
    """Fetch anything `cog.yaml` did not already bake into the image."""
    for filename in HF_MODEL_FILES:
        if not (REPO_ROOT / filename).exists():
            print(f"[setup] Downloading missing checkpoint: {filename}")
            download_model(filename)
    if not (REPO_ROOT / LLM_LOCAL_DIR / "config.json").exists():
        print("[setup] Downloading missing Qwen3-0.6B config")
        download_llm()


def start_bpy_server() -> subprocess.Popen:
    """Launch bpy_server.py in its own process group so it can be torn down."""
    popen_kwargs = dict(args=[sys.executable, "bpy_server.py"], cwd=str(REPO_ROOT))
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(**popen_kwargs)
    print(f"[setup] bpy_server.py started (pid={proc.pid})")

    def cleanup() -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    atexit.register(cleanup)
    return proc


def wait_for_bpy_server(proc: subprocess.Popen, timeout: int = BPY_STARTUP_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"bpy_server exited during startup (code {proc.returncode})")
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[setup] bpy_server is ready")
            return
        except requests.RequestException:
            if time.time() > deadline:
                raise RuntimeError(f"bpy_server failed to start within {timeout}s")
            time.sleep(0.5)


def post_bpy_payload(endpoint: str, payload):
    """Hand a payload to the bpy server via a temp file (payloads can be large)."""
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False
        ) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes({"payload_path": payload_path}),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None and os.path.exists(payload_path):
            try:
                os.remove(payload_path)
            except OSError:
                pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download any missing weights, start the bpy server, load the model."""
        ensure_weights()

        self._bpy_proc = start_bpy_server()
        wait_for_bpy_server(self._bpy_proc)

        print(f"[setup] Loading TokenRig checkpoint: {MODEL_CKPT}")
        self.model = get_model(MODEL_CKPT)
        assert self.model.tokenizer_config is not None
        self.tokenizer = get_tokenizer(**self.model.tokenizer_config)
        self.transform = Transform.parse(**self.model.transform_config["predict_transform"])
        self._workdir: Optional[pathlib.Path] = None
        print("[setup] Ready")

    def predict(
        self,
        mesh: Path = Input(
            description="Input 3D mesh to rig (.glb, .obj or .fbx).",
        ),
        use_skeleton: bool = Input(
            default=False,
            description=(
                "Keep the skeleton that already exists in the input file and only "
                "generate skinning weights. Ignored if the input has no skeleton."
            ),
        ),
        use_transfer: bool = Input(
            default=True,
            description=(
                "Transfer the generated rig back onto the original mesh, preserving "
                "its textures and scale. Disable to export the normalized mesh."
            ),
        ),
        use_postprocess: bool = Input(
            default=False,
            description="Apply voxel-based skinning post-processing to suppress weight bleeding.",
        ),
        top_k: int = Input(default=5, ge=1, le=200, description="Top-k sampling."),
        top_p: float = Input(default=0.95, ge=0.1, le=1.0, description="Top-p (nucleus) sampling."),
        temperature: float = Input(default=1.0, ge=0.1, le=2.0, description="Sampling temperature."),
        repetition_penalty: float = Input(
            default=2.0, ge=0.5, le=3.0, description="Repetition penalty."
        ),
        num_beams: int = Input(
            default=10, ge=1, le=20, description="Number of beams for beam search."
        ),
        seed: Optional[int] = Input(
            default=None, description="Random seed. Leave blank for a random seed."
        ),
    ) -> Path:
        """Generate a skeleton and skinning weights, and return a rigged GLB."""
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        seed_everything(seed)

        source = pathlib.Path(str(mesh))
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported input '{source.name}'. Supported extensions: "
                f"{', '.join(sorted(SUPPORTED_EXT))}."
            )

        # Blender holds file handles for the duration of a run, so give every
        # prediction a clean directory and drop the previous one.
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = pathlib.Path(tempfile.mkdtemp(prefix="tokenrig_"))
        input_path = self._workdir / f"input{suffix}"
        shutil.copyfile(source, input_path)
        output_path = self._workdir / "rigged.glb"

        # The server holds all Blender state; a crash on one mesh must not take
        # down every later prediction.
        if self._bpy_proc.poll() is not None:
            print("[predict] bpy_server is not running, restarting it")
            self._bpy_proc = start_bpy_server()
            wait_for_bpy_server(self._bpy_proc)

        batch = self._load_batch(input_path)

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1,
            num_beams=int(num_beams),
            do_sample=True,
        )

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = [batch["skeleton_tokens"][0][mask].cpu().numpy()]
        else:
            if use_skeleton:
                print("[predict] No skeleton found in the input, generating one instead")
            skeleton_tokens = None

        preds: List[TokenRigResult] = self.model.predict_step(
            batch,
            skeleton_tokens=skeleton_tokens,
            make_asset=True,
        )["results"]

        asset = preds[0].asset
        if asset is None:
            raise RuntimeError("The model did not produce a rigged asset for this mesh.")

        if use_postprocess:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        if use_transfer:
            result = post_bpy_payload(
                "transfer",
                dict(
                    source_asset=asset,
                    target_path=asset.path,
                    export_path=str(output_path),
                    group_per_vertex=4,
                ),
            )
        else:
            result = post_bpy_payload(
                "export",
                dict(asset=asset, filepath=str(output_path), group_per_vertex=4),
            )

        if result != "ok":
            raise RuntimeError(f"Export failed: {result}")
        if not output_path.exists():
            raise RuntimeError(f"Export reported success but {output_path} is missing.")

        return Path(output_path)

    def _load_batch(self, input_path: pathlib.Path) -> dict:
        """Run one mesh through the predict transform and move it onto the GPU."""
        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
            datapath={
                "data_name": None,
                "loader": "bpy_server",
                "filepaths": {"articulation": [str(input_path)]},
            },
        ).split_by_cls()

        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=self.transform,
            tokenizer=self.tokenizer,
            process_fn=self.model._process_fn,
        )

        dataloader = module.predict_dataloader()["articulation"]
        batch = next(iter(dataloader))
        return {
            k: v.to("cuda") if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
