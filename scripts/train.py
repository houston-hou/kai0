import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb


import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


print("device_count", jax.device_count())
print("devices", jax.devices())
print("process_count", jax.process_count())

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def heldout_eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> _model.Actions:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, _ = batch
    return model.sample_actions(rng, observation, num_steps=config.heldout_eval_num_steps)


def _make_output_transform(data_config: _config.DataConfig):
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def _to_host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _decode_eval_actions(
    output_transform,
    observation: _model.Observation,
    actions: np.ndarray,
) -> np.ndarray:
    decoded = output_transform(
        {
            "state": _to_host_array(observation.state),
            "actions": np.asarray(actions),
        }
    )
    return np.asarray(decoded["actions"], dtype=np.float32)


def _make_heldout_plot(true_actions: np.ndarray, pred_actions: np.ndarray, *, max_timesteps: int):
    import matplotlib.pyplot as plt

    names = [
        "left_0",
        "left_1",
        "left_2",
        "left_3",
        "left_4",
        "left_5",
        "left_gripper",
        "right_0",
        "right_1",
        "right_2",
        "right_3",
        "right_4",
        "right_5",
        "right_gripper",
    ]
    dims = min(true_actions.shape[-1], pred_actions.shape[-1], len(names))
    true_flat = true_actions.reshape(-1, true_actions.shape[-1])
    pred_flat = pred_actions.reshape(-1, pred_actions.shape[-1])
    timesteps = min(max(1, max_timesteps), true_flat.shape[0], pred_flat.shape[0])
    true_flat = true_flat[:timesteps]
    pred_flat = pred_flat[:timesteps]

    fig, axes = plt.subplots(7, 2, figsize=(18, 22), sharex=True)
    axes = axes.ravel()
    xs = np.arange(timesteps)
    for dim in range(dims):
        ax = axes[dim]
        ax.plot(xs, true_flat[:, dim], color="#1f77b4", linewidth=1.0, alpha=0.85)
        ax.plot(xs, pred_flat[:, dim], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.set_title(names[dim])
        ax.grid(True, alpha=0.25)
    axes[0].plot([], [], color="#1f77b4", label="expert action")
    axes[0].plot([], [], color="#d62728", linestyle="--", label="model action")
    axes[0].legend(loc="best")
    fig.tight_layout()
    return fig


def log_heldout_eval(
    *,
    step: int,
    config: _config.TrainConfig,
    output_transform,
    eval_batch: tuple[_model.Observation, _model.Actions],
    pred_actions: np.ndarray,
) -> dict[str, float]:
    observation, true_actions_raw = eval_batch
    true_actions = _decode_eval_actions(output_transform, observation, _to_host_array(true_actions_raw))
    pred_actions = _decode_eval_actions(output_transform, observation, pred_actions)

    samples = min(config.heldout_eval_samples, true_actions.shape[0], pred_actions.shape[0])
    horizon = config.heldout_eval_horizon or min(true_actions.shape[1], pred_actions.shape[1])
    horizon = min(horizon, true_actions.shape[1], pred_actions.shape[1])
    dims = min(true_actions.shape[-1], pred_actions.shape[-1])
    true_slice = true_actions[:samples, :horizon, :dims]
    pred_slice = pred_actions[:samples, :horizon, :dims]
    error = pred_slice - true_slice
    mae_by_dim = np.mean(np.abs(error), axis=(0, 1))
    rmse_by_dim = np.sqrt(np.mean(np.square(error), axis=(0, 1)))

    payload: dict[str, Any] = {
        "heldout_action/mae": float(np.mean(mae_by_dim)),
        "heldout_action/rmse": float(np.mean(rmse_by_dim)),
        "heldout_action/max_abs_error": float(np.max(np.abs(error))),
    }
    for dim, (mae, rmse) in enumerate(zip(mae_by_dim, rmse_by_dim, strict=False)):
        payload[f"heldout_action_dim/{dim:02d}_mae"] = float(mae)
        payload[f"heldout_action_dim/{dim:02d}_rmse"] = float(rmse)

    fig = _make_heldout_plot(true_slice, pred_slice, max_timesteps=config.heldout_eval_compare_timesteps)
    payload["heldout_action/trajectory_overlay"] = wandb.Image(fig)
    wandb.log(payload, step=step)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return {key: value for key, value in payload.items() if isinstance(value, float)}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    heldout_eval_batch = None
    heldout_output_transform = None
    if config.heldout_eval_enabled:
        heldout_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=1,
        )
        heldout_eval_batch = next(iter(heldout_loader))
        heldout_output_transform = _make_output_transform(heldout_loader.data_config())
        logging.info(f"Initialized held-out eval batch:\n{training_utils.array_tree_to_info(heldout_eval_batch)}")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pheldout_eval_step = jax.jit(
        functools.partial(heldout_eval_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    heldout_eval_rng = jax.random.fold_in(train_rng, 12345)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if config.heldout_eval_enabled and heldout_eval_batch is not None and heldout_output_transform is not None:
                with sharding.set_mesh(mesh):
                    pred_actions = pheldout_eval_step(heldout_eval_rng, train_state, heldout_eval_batch)
                eval_info = log_heldout_eval(
                    step=step,
                    config=config,
                    output_transform=heldout_output_transform,
                    eval_batch=heldout_eval_batch,
                    pred_actions=_to_host_array(pred_actions),
                )
                eval_str = ", ".join(f"{k}={v:.4f}" for k, v in eval_info.items())
                pbar.write(f"Step {step} heldout: {eval_str}")
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
