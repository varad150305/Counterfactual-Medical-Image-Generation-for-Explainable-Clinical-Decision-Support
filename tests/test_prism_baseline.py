"""Phase 3 tests: reconstructed PRISM components + the clean inference interface.

Uses the tiny dummy-weight SD pipeline (hf-internal-testing/tiny-stable-diffusion-pipe,
same fixture as tests/test_environment.py) throughout — fast, offline-after-first-cache,
and exercises the real code paths (tokenization, attention-processor registration,
DDIM inversion math, attention control, CLIP scoring) without needing PRISM's
3.44GB checkpoint or a multi-minute real generation. Real-checkpoint /
real-image validation is documented separately in
docs/PHASE_3_PRISM_BASELINE.md, not re-run inside the automated suite.
"""
import numpy as np
import pytest
import torch
from PIL import Image

TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"


@pytest.fixture(scope="module")
def tiny_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False, steps_offset=1,
    )
    pipe = StableDiffusionPipeline.from_pretrained(TINY_SD, scheduler=scheduler, safety_checker=None)
    pipe.to("cpu")
    return pipe


@pytest.fixture()
def sample_image_path(tmp_path):
    path = tmp_path / "sample.png"
    rng = np.random.RandomState(0)
    Image.fromarray((rng.rand(64, 64, 3) * 255).astype("uint8")).save(path)
    return str(path)


class TestBaselineConfigFile:
    def test_prism_baseline_yaml_numeric_keys_are_numbers(self):
        """Regression: `early_stop_epsilon: 1e-5` loaded as the *string*
        "1e-5" under PyYAML (YAML 1.1 needs a decimal point) and crashed the
        first real run at `loss.item() < epsilon + ...`. Unit tests that pass
        Python dicts never exercise the YAML loader, so check the shipped
        config file itself."""
        from pathlib import Path

        from src.utils.config import load_config

        cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "prism_baseline.yaml"))
        editing = cfg["editing"]
        for key in (
            "image_size", "num_ddim_steps", "num_inner_steps", "guidance_scale", "self_replace_steps",
            "cross_replace_steps", "edit_word_weight", "early_stop_epsilon", "seed",
        ):
            assert isinstance(editing[key], (int, float)) and not isinstance(editing[key], bool), (
                f"editing.{key}={editing[key]!r} is {type(editing[key]).__name__}, expected a number"
            )
        assert isinstance(editing["local_blend"], bool)

    def test_string_epsilon_from_config_does_not_crash(self, sample_image_path):
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {"num_ddim_steps": 2, "num_inner_steps": 1, "early_stop_epsilon": "1e-5", "seed": 0},
            "evaluation": {"compute_clip_metrics": False},
        }
        result = PrismCounterfactualGenerator(config).generate(sample_image_path, "Cardiomegaly")
        assert result.counterfactual_image is not None


class TestImports:
    def test_prism_modules_import(self):
        from src.models.prism import attention_control, clip_similarity, null_inversion, ptp_utils, token_alignment  # noqa: F401

    def test_inference_interface_imports(self):
        from src.inference.prism_baseline import CounterfactualResult, PrismCounterfactualGenerator, build_caption  # noqa: F401


class TestPromptConstruction:
    def test_add_operation_caption(self):
        from src.inference.prism_baseline import build_caption

        assert build_caption("Cardiomegaly") == "Chest X-ray showing Cardiomegaly"

    def test_no_finding_caption(self):
        from src.inference.prism_baseline import NO_FINDING_CAPTION, build_caption

        assert build_caption("No Finding") == NO_FINDING_CAPTION
        assert build_caption(None) == NO_FINDING_CAPTION

    def test_build_caption_pneumonia(self):
        from src.inference.prism_baseline import build_caption

        assert build_caption("Pneumonia") == "Chest X-ray showing Pneumonia"


class TestTokenAlignment:
    def test_get_word_inds_finds_multi_token_word(self, tiny_pipe):
        from src.models.prism.token_alignment import get_word_inds

        inds = get_word_inds(tiny_pipe.tokenizer, "chest x-ray showing cardiomegaly", "cardiomegaly")
        assert len(inds) >= 1

    def test_refinement_mapper_shapes(self, tiny_pipe):
        from src.models.prism.token_alignment import get_refinement_mapper

        mapper, alphas = get_refinement_mapper(
            tiny_pipe.tokenizer, "normal chest x-ray with no significant findings", "chest x-ray showing cardiomegaly"
        )
        assert mapper.shape == (tiny_pipe.tokenizer.model_max_length,)
        assert alphas.shape == (tiny_pipe.tokenizer.model_max_length, 1)
        assert torch.all((alphas == 0) | (alphas == 1))


class TestControllerInvariants:
    def test_identical_prompts_mapper_is_identity(self, tiny_pipe):
        """Regression: EOS/padding positions used to all map to the original's
        single EOS index, copying one token's attention into ~70 padding
        columns (attention rows no longer summed to 1 -> global brightening).
        With identical prompts the mapper must be the identity map."""
        from src.models.prism.token_alignment import get_refinement_mapper

        text = "chest x-ray with no significant findings"
        mapper, alphas = get_refinement_mapper(tiny_pipe.tokenizer, text, text)
        assert torch.equal(mapper, torch.arange(tiny_pipe.tokenizer.model_max_length))
        assert torch.all(alphas == 1.0)

    def test_tail_positions_map_positionally_for_unequal_lengths(self, tiny_pipe):
        from src.models.prism.token_alignment import get_refinement_mapper

        tok = tiny_pipe.tokenizer
        orig, edit = "chest x-ray with no significant findings", "chest x-ray showing cardiomegaly"
        mapper, _ = get_refinement_mapper(tok, orig, edit)
        n_orig = len(tok.encode(orig, add_special_tokens=False)) + 2
        n_edit = len(tok.encode(edit, add_special_tokens=False)) + 2
        # edited EOS aligns to original EOS, and the following pad positions advance in step
        assert mapper[n_edit - 1].item() == n_orig - 1
        assert mapper[n_edit].item() == min(n_orig, tok.model_max_length - 1)

    def test_identical_prompts_controller_is_noop_end_to_end(self, tiny_pipe):
        """The strongest invariant: editing prompt A -> A through the full
        controller must leave the 'edited' branch equal to the base branch."""
        from src.models.prism import ptp_utils
        from src.models.prism.attention_control import make_controller

        prompt = "chest x-ray with no significant findings"
        controller = make_controller(
            tiny_pipe.tokenizer, prompt, prompt, num_steps=3, cross_replace_frac=0.8, self_replace_frac=0.6,
            blend_word=None, edit_word=None, edit_word_weight=1.0, device="cpu",
        )
        gen = torch.Generator("cpu").manual_seed(0)
        size = tiny_pipe.unet.config.sample_size
        latent = torch.randn((1, tiny_pipe.unet.config.in_channels, size, size), generator=gen)
        tok = tiny_pipe.tokenizer
        uncond = tiny_pipe.text_encoder(
            tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
        )[0]
        images, _ = ptp_utils.run_and_display(
            tiny_pipe, [prompt, prompt], controller, latent=latent, uncond_embeddings=uncond,
            num_inference_steps=3, guidance_scale=7.5,
        )
        diff = np.abs(images[0].astype(int) - images[1].astype(int))
        assert diff.max() <= 1, f"identity edit changed pixels by up to {diff.max()}/255 (mean {diff.mean():.2f})"


class TestLocalBlend:
    @staticmethod
    def _run(tiny_pipe, orig, edit, blend_word):
        from src.models.prism import ptp_utils
        from src.models.prism.attention_control import make_controller

        controller = make_controller(
            tiny_pipe.tokenizer, orig, edit, num_steps=3, cross_replace_frac=0.8, self_replace_frac=0.5,
            blend_word=blend_word, edit_word=None, edit_word_weight=1.0, device="cpu",
        )
        gen = torch.Generator("cpu").manual_seed(0)
        size = tiny_pipe.unet.config.sample_size
        latent = torch.randn((1, tiny_pipe.unet.config.in_channels, size, size), generator=gen)
        tok = tiny_pipe.tokenizer
        uncond = tiny_pipe.text_encoder(
            tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
        )[0]
        ptp_utils.run_and_display(
            tiny_pipe, [orig, edit], controller, latent=latent, uncond_embeddings=uncond,
            num_inference_steps=3, guidance_scale=7.5,
        )
        return controller

    def test_token_mask_marks_target_word_tokens_per_prompt(self, tiny_pipe):
        """Regression: the mask was once built as (equalizer - 1).abs() with
        equalizer value 1.0, i.e. always all-zero, so LocalBlend silently
        disabled itself and never ran. Rows are per-prompt: for an 'add' the
        word is only in the edited prompt (row 1)."""
        from src.models.prism.attention_control import LocalBlend
        from src.models.prism.token_alignment import get_word_inds

        orig, edit = "normal chest x-ray with no significant findings", "chest x-ray showing cardiomegaly"
        lb = LocalBlend(tiny_pipe.tokenizer, [orig, edit], ["cardiomegaly"])
        assert lb.token_mask is not None
        assert lb.token_mask[0].sum() == 0
        expected = get_word_inds(tiny_pipe.tokenizer, edit, "cardiomegaly")
        assert sorted(lb.token_mask[1].nonzero().flatten().tolist()) == sorted(expected)

    def test_remove_case_mask_comes_from_original_prompt(self, tiny_pipe):
        """For 'remove' the target word exists only in the ORIGINAL caption; the
        blend used to be built from the edited caption alone and disabled itself."""
        from src.models.prism.attention_control import LocalBlend

        orig, edit = "chest x-ray showing cardiomegaly", "normal chest x-ray with no significant findings"
        lb = LocalBlend(tiny_pipe.tokenizer, [orig, edit], ["cardiomegaly"])
        assert lb.token_mask is not None
        assert lb.token_mask[0].sum() > 0 and lb.token_mask[1].sum() == 0

    def test_word_in_neither_prompt_disables_blend_instead_of_masking_everything(self, tiny_pipe):
        from src.models.prism.attention_control import LocalBlend

        lb = LocalBlend(
            tiny_pipe.tokenizer, ["normal chest x-ray", "chest x-ray with no findings"], ["pneumonia"]
        )
        assert lb.token_mask is None

    def test_blend_runs_for_add(self, tiny_pipe):
        c = self._run(
            tiny_pipe, "chest x-ray with no significant findings", "chest x-ray showing cardiomegaly", "cardiomegaly"
        )
        assert c.local_blend.last_mask_fraction is not None, "LocalBlend never produced a mask"
        assert 0.0 <= c.local_blend.last_mask_fraction <= 1.0

    def test_blend_runs_for_remove(self, tiny_pipe):
        c = self._run(
            tiny_pipe, "chest x-ray showing cardiomegaly", "chest x-ray with no significant findings", "cardiomegaly"
        )
        assert c.local_blend.last_mask_fraction is not None, "LocalBlend disabled itself for 'remove'"
        assert 0.0 <= c.local_blend.last_mask_fraction <= 1.0


class TestAttentionControl:
    def test_register_attention_control_counts_layers(self, tiny_pipe):
        from src.models.prism.attention_control import AttentionStore
        from src.models.prism.ptp_utils import register_attention_control

        controller = AttentionStore()
        count = register_attention_control(tiny_pipe.unet, controller)
        assert count > 0
        assert controller.num_att_layers == count

    def test_make_controller_builds_editing_config(self, tiny_pipe):
        from src.models.prism.attention_control import AttentionControlEdit, make_controller

        controller = make_controller(
            tiny_pipe.tokenizer, "normal chest x-ray with no significant findings", "chest x-ray showing cardiomegaly",
            num_steps=4, cross_replace_frac=0.8, self_replace_frac=0.5,
            blend_word="cardiomegaly", edit_word="cardiomegaly", edit_word_weight=2.0, device="cpu",
        )
        assert isinstance(controller, AttentionControlEdit)
        assert controller.local_blend is not None
        assert controller.num_self_replace == (0, 2)

    def test_make_controller_without_edit_word_weight_skips_reweight(self, tiny_pipe):
        from src.models.prism.attention_control import AttentionRefine, make_controller

        controller = make_controller(
            tiny_pipe.tokenizer, "a", "b", num_steps=4, cross_replace_frac=0.8, self_replace_frac=0.5,
            blend_word=None, edit_word=None, edit_word_weight=1.0, device="cpu",
        )
        assert isinstance(controller, AttentionRefine)


class TestNullInversion:
    def test_invert_returns_expected_shapes(self, tiny_pipe, sample_image_path):
        from src.models.prism.null_inversion import NullInversion

        inv = NullInversion(tiny_pipe, "cpu", num_ddim_steps=2, guidance_scale=7.5)
        (image_gt, _), x_t, uncond_embeddings = inv.invert(sample_image_path, "a chest x-ray", num_inner_steps=1)

        assert image_gt.shape == (64, 64, 3)
        assert x_t.shape[0] == 1
        assert not torch.isnan(x_t).any()
        assert len(uncond_embeddings) == 2

    def test_invert_missing_image_raises(self, tiny_pipe, tmp_path):
        from src.models.prism.null_inversion import NullInversion

        inv = NullInversion(tiny_pipe, "cpu", num_ddim_steps=2)
        with pytest.raises(FileNotFoundError):
            inv.invert(str(tmp_path / "nope.png"), "a chest x-ray", num_inner_steps=1)


class TestClipSimilarity:
    def test_call_returns_four_metrics(self):
        from src.models.prism.clip_similarity import ClipSimilarity

        clip_sim = ClipSimilarity(device="cpu")
        img1 = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        s0, s1, sdir, simg = clip_sim([img1], [img2], ["a dark image"], ["a bright image"])
        for t in (s0, s1, sdir, simg):
            assert torch.is_tensor(t)
        assert -1.0 <= float(s0) <= 1.0
        assert -1.0 <= float(simg[0]) <= 1.0

    def test_pred_consistency_returns_bool(self):
        from src.models.prism.clip_similarity import ClipSimilarity

        clip_sim = ClipSimilarity(device="cpu")
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        result = clip_sim.pred_consistency(img, "no finding", "cardiomegaly")
        assert isinstance(result, bool)


class TestInferenceInterface:
    @pytest.fixture()
    def generator(self, sample_image_path):
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {
                "num_ddim_steps": 2, "num_inner_steps": 1, "guidance_scale": 7.5,
                "self_replace_steps": 0.5, "cross_replace_steps": 0.8, "edit_word_weight": 2.0,
                "local_blend": True, "seed": 0,
            },
            "evaluation": {"compute_clip_metrics": True},
        }
        return PrismCounterfactualGenerator(config)

    def test_generate_add_operation(self, generator, sample_image_path):
        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")

        assert result.original_caption == "Normal chest X-ray with no significant findings"
        assert result.edited_caption == "Chest X-ray showing Cardiomegaly"
        assert result.target_pathology == "Cardiomegaly"
        assert result.operation == "add"

    def test_generate_remove_operation(self, generator, sample_image_path):
        result = generator.generate(sample_image_path, "Pneumonia", operation="remove")

        assert result.original_caption == "Chest X-ray showing Pneumonia"
        assert result.edited_caption == "Normal chest X-ray with no significant findings"

    def test_generate_invalid_operation_raises(self, generator, sample_image_path):
        with pytest.raises(ValueError):
            generator.generate(sample_image_path, "Cardiomegaly", operation="modify")

    def test_generate_missing_image_raises(self, generator, tmp_path):
        with pytest.raises(FileNotFoundError):
            generator.generate(str(tmp_path / "nope.png"), "Cardiomegaly")

    def test_output_validity(self, generator, sample_image_path):
        """[output validation] dimensions, pixel range, no NaN, valid image."""
        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")

        for img in (result.original_image, result.reconstructed_image, result.counterfactual_image):
            assert isinstance(img, Image.Image)
            arr = np.array(img)
            assert arr.shape == (64, 64, 3)
            assert arr.dtype == np.uint8
            assert not np.isnan(arr.astype(float)).any()

        assert result.difference_map.shape == (64, 64)
        assert result.difference_map.dtype == np.uint8

    def test_metrics_present_and_in_range(self, generator, sample_image_path):
        """[image-image similarity / image-text similarity if supported]"""
        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")

        for key in (
            "clip_image_text_similarity_original", "clip_image_text_similarity_edited",
            "clip_directional_similarity", "clip_image_image_similarity",
        ):
            assert key in result.metrics
            assert -1.01 <= result.metrics[key] <= 1.01

    def test_inference_info_has_timings(self, generator, sample_image_path):
        """[generation time]"""
        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")

        assert "timings" in result.inference_info
        assert result.inference_info["timings"]["inversion_seconds"] >= 0
        assert result.inference_info["timings"]["editing_seconds"] >= 0
        assert result.inference_info["total_seconds"] >= 0

    def test_result_save_writes_expected_files(self, generator, sample_image_path, tmp_path):
        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")
        out_dir = result.save(str(tmp_path / "out"))

        from pathlib import Path

        out_path = Path(out_dir)
        for fname in ("original.png", "reconstructed.png", "counterfactual.png", "difference_map.png", "metadata.json"):
            assert (out_path / fname).exists()

    def test_image_size_config_is_consistent_across_inversion_and_editing(self, sample_image_path):
        """Regression: editing.image_size used to reach NullInversion but not
        run_and_display, so the edit pass fell back to the UNet's default
        resolution and mismatched the inverted latent. Every output must
        come out at the configured size, and the inverted latent and edit
        pass must agree on it."""
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {
                "image_size": 32, "num_ddim_steps": 2, "num_inner_steps": 1,
                "self_replace_steps": 0.5, "cross_replace_steps": 0.8, "seed": 0,
            },
            "evaluation": {"compute_clip_metrics": False},
        }
        generator = PrismCounterfactualGenerator(config)
        assert generator.inverter.image_size == 32

        result = generator.generate(sample_image_path, "Cardiomegaly", operation="add")
        for img in (result.original_image, result.reconstructed_image, result.counterfactual_image):
            assert img.size == (32, 32)
        assert result.difference_map.shape == (32, 32)

    def test_inversion_cache_hit_reproduces_result(self, sample_image_path, tmp_path):
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {
                "num_ddim_steps": 2, "num_inner_steps": 1, "self_replace_steps": 0.5,
                "seed": 0, "inversion_cache_dir": str(tmp_path / "cache"),
            },
            "evaluation": {"compute_clip_metrics": False},
        }
        gen = PrismCounterfactualGenerator(config)
        first = gen.generate(sample_image_path, "Cardiomegaly")
        second = gen.generate(sample_image_path, "Cardiomegaly")

        assert first.inference_info["inversion_cached"] is False
        assert second.inference_info["inversion_cached"] is True
        assert len(list((tmp_path / "cache").glob("inversion_*.pt"))) == 1
        assert np.array_equal(np.array(first.counterfactual_image), np.array(second.counterfactual_image))

    def test_num_ddim_steps_override_keeps_scheduler_in_sync(self, sample_image_path):
        """Regression: a per-call num_ddim_steps override changed the inverter's
        step count but not the scheduler timesteps."""
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {"num_ddim_steps": 4, "num_inner_steps": 1, "seed": 0},
            "evaluation": {"compute_clip_metrics": False},
        }
        gen = PrismCounterfactualGenerator(config)
        gen.generate(sample_image_path, "Cardiomegaly", num_ddim_steps=2)
        assert len(gen.pipe.scheduler.timesteps) == 2

    def test_strict_unet_loader_roundtrip_and_rejects_incompatible(self, tiny_pipe, tmp_path):
        from safetensors.torch import save_file

        from src.inference.prism_baseline import PrismCounterfactualGenerator

        loader = PrismCounterfactualGenerator._load_unet_weights_strict
        state = {k: v.clone() for k, v in tiny_pipe.unet.state_dict().items()}

        good = tmp_path / "good.safetensors"
        save_file({k: v + 1.0 if v.is_floating_point() else v for k, v in state.items()}, str(good))
        loader(tiny_pipe.unet, str(good))
        first_key = next(k for k, v in state.items() if v.is_floating_point())
        assert torch.allclose(tiny_pipe.unet.state_dict()[first_key], state[first_key] + 1.0)

        bad = tmp_path / "bad.safetensors"
        save_file({k: v for i, (k, v) in enumerate(state.items()) if i > 0}, str(bad))
        with pytest.raises(RuntimeError, match="incompatible with the UNet architecture"):
            loader(tiny_pipe.unet, str(bad))

        # restore original weights so the module-scoped fixture isn't left perturbed
        restore = tmp_path / "restore.safetensors"
        save_file(state, str(restore))
        loader(tiny_pipe.unet, str(restore))

    def test_missing_finetuned_checkpoint_raises_clear_error(self, tmp_path):
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        config = {
            "model": {
                "ldm_type": "finetuned_chexpert", "base_model_id": TINY_SD,
                "finetuned_checkpoint_path": str(tmp_path / "nope.safetensors"), "device": "cpu",
            },
            "editing": {"num_ddim_steps": 2},
            "evaluation": {"compute_clip_metrics": False},
        }
        with pytest.raises(FileNotFoundError, match="finetuned_checkpoint_path"):
            PrismCounterfactualGenerator(config)
