"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
"""
DEBUG_MODE: bool = True
import logging

from .solver import *
from .model import *
from .module import *
from .utils import isinstance_str


class CacheBus:
    """A Bus class for overall control."""

    def __init__(self):
        # == Tensor Caching ==
        self.prev_epsilon_guided = [None, None]
        self.prev_epsilon = None
        self.prev_interp = None
        self.prev_f = [None, None]
        self.lagrange_x0 = []
        self.prev_x = [None, None]

        # == Estimator ==
        self.pred_m_m_1 = None
        self.taylor_m_m_1 = None
        self.temporal_score = None

        # == Control Signals ==
        self.skip_this_step = False

        # == Control Variables ==
        self.step = 0
        self.cons_skip = 0
        self.cons_prune = 0
        self.lagrange_step = []
        self.last_skip_step = 0  # align with step in cache bus
        self.ind_step = None
        self.c_step = 1  # doesn't really matter

        # == Optimizations ==
        self.m_a = None
        self.u_a = None

        # == Logs ==
        # self.model_outputs = {}
        # self.model_outputs_change = {}
        self.pred_error_list = []
        self.taylor_error_list = []
        self.abs_momentum_list = []
        self.rel_momentum_list = []
        self.skipping_path = []


def hook_tome_model(model: torch.nn.Module):
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def reset_cache(model: torch.nn.Module):
    if hasattr(model, "unet"):
        diffusion_model = model.unet
    elif hasattr(model, "transformer"):
        diffusion_model = model.transformer
    else:
        raise NotImplementedError

    # reset bus
    bus = diffusion_model._cache_bus

    # == Tensor Caching ==
    bus.prev_epsilon = None
    bus.prev_epsilon_guided = [None, None]
    bus.prev_f = [None, None]

    # == Estimator ==
    bus.pred_m_m_1 = None
    bus.taylor_m_m_1 = None
    bus.temporal_score = None

    # == Control Signals ==
    bus.skip_this_step = False

    # == Control Variables ==
    bus.step = 0
    bus.cons_skip = 0
    bus.last_skip_step = 0  # align with step in cache bus
    bus.ind_step = None
    bus.c_step = 1  # doesn't really matter

    # == Optimizations ==
    bus.m_a = None
    bus.u_a = None

    # == Logs ==
    # bus.model_outputs = {}
    # bus.model_outputs_change = {}
    bus.pred_error_list = []
    bus.taylor_error_list = []
    bus.abs_momentum_list = []
    bus.rel_momentum_list = []
    bus.skipping_path = []


    # re-patch
    index = 0
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "ToMeBlock"):
            index += 1
        elif isinstance_str(module, "FluxTransformerBlock"):
            index += 1
        elif isinstance_str(module, "FluxSingleTransformerBlock"):
            index += 1
    print(f"Reset cache for {index} BasicTransformerBlock")

    return model


def apply_patch(
        model: torch.nn.Module,

        denominator: int = 2,
        modular: tuple = (1,),
        acc_range: Tuple[int, int] = (10, 47),
        interp_mode: Literal["psi", "x_0"] = "psi",
        caching_mode: Literal["reuse_all", "interp_all", "reuse_interp"] = "reuse_interp",

        lagrange_term: int = 0,
        lagrange_int: int = None,
        lagrange_step: int = None,

        max_interval: int = 4,

        test_skip_path: List[int] = None,
):
    # == merging preparation ==
    global DEBUG_MODE
    if DEBUG_MODE: print('Start with \033[95mDEBUG\033[0m mode')
    print('\033[94mApplying \033[93m⚡Zeus\033[0m Patches\033[0m')

    acc_start = max(acc_range[0], 3) # we leveraged third order
    for m in modular:
        assert m < denominator, "skip ratio should less than 1"

    remove_patch(model)
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        raise RuntimeError("Only support huggingface diffuser")
    else:
        if hasattr(model, "unet"):
            diffusion_model = model.unet
        elif hasattr(model, "transformer"):
            diffusion_model = model.transformer
        else:
            raise NotImplementedError

    solver = model.scheduler

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "generator": None,

            # hyperparameters
            "denominator": denominator,
            "modular": modular,
            "acc_range": (acc_start, acc_range[-1]),
            "interp_mode": interp_mode,
            "caching_mode": caching_mode,

            # lagrangian interpolation
            "lagrange_term": lagrange_term,
            "lagrange_int": lagrange_int,
            "lagrange_step": lagrange_step,

            # maximum configuration
            "max_interval": max_interval,

            "test_skip_path": test_skip_path,
        }
    }

    if hasattr(model, "unet"):
        hook_tome_model(diffusion_model)
        diffusion_model.__class__ = patch_unet(diffusion_model.__class__)
    elif hasattr(model, "transformer"):
        diffusion_model.__class__ = patch_transformer(diffusion_model.__class__)
    else:
        raise NotImplementedError

    diffusion_model._cache_bus = CacheBus()
    diffusion_model._cache_bus._tome_info = diffusion_model._tome_info

    solver.__class__ = patch_solver(solver.__class__)
    solver._cache_bus = diffusion_model._cache_bus

    # lagrangian interpolation configuration
    if lagrange_term != 0:
        assert lagrange_step % lagrange_int == 0, "For lagrangian, please make sure (lagrangian_step % lagrangian_interval == 0)"
        diffusion_model._cache_bus.lagrange_step = [None] * lagrange_term
        diffusion_model._cache_bus.lagrange_x0 = [None] * lagrange_term

    print(f"Applied patch.")
    return model


def remove_patch(model: torch.nn.Module):
    # For diffusers
    if hasattr(model, "unet"):
        model = model.unet
    elif hasattr(model, "transformer"):
        model = model.transformer
    else:
        raise NotImplementedError

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model


def get_logged_feature_maps(model: torch.nn.Module, file_name: str = "outputs/model_outputs.npz"):
    logging.debug(f"\033[96mLogging Feature Map\033[0m")
    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in
                          model._bus.model_outputs.items()}
    np.savez(file_name, **numpy_feature_maps)

    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in
                          model._bus.model_outputs_change.items()}
    np.savez("outputs/model_outputs_change.npz", **numpy_feature_maps)

