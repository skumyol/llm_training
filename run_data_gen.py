"""
Data generation pipeline entry point.

Usage:
    python run_data_gen.py --config configs/data_gen.yaml
    python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 5
"""
import argparse
import json
import os
import queue
import random
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data_gen.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Use mock LLM calls (no API key needed, for testing)")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Override n_episodes from config")
    p.add_argument("--stage", choices=["generate", "validate", "package", "split", "all"],
                   default="all")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_episodes is not None:
        cfg["generation"]["n_episodes"] = args.n_episodes

    os.makedirs(cfg["output"]["raw_episodes_dir"],    exist_ok=True)
    os.makedirs(cfg["output"]["validated_turns_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["counterfactuals_dir"], exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/npc_profiles", exist_ok=True)

    if not args.no_mlflow:
        import mlflow
        from src.mlflow_utils import setup_mlflow
        setup_mlflow(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    if args.stage in ("generate", "all"):
        _run_generation(cfg, args)

    if args.stage in ("validate", "all"):
        _run_validation(cfg)

    if args.stage in ("package", "all"):
        _run_packaging(cfg, args)

    if args.stage in ("split", "all"):
        _run_split(cfg)


def _run_generation(cfg: dict, args) -> None:
    from src.data_gen.scenario_bank import ScenarioBank
    from src.data_gen.state_init import build_npc_profile, ROLE_PROFILES
    from src.data_gen.episode_planner import plan_episode
    from src.data_gen.validator import Validator
    from src.data_gen.world_context import WorldContext

    scenario_weights = cfg["scenario"].get("scenario_type_weights", {})
    bank = ScenarioBank(cfg["scenario"]["bank_dir"], seed=cfg.get("seed", 42))
    
    # Load World Context if available
    world_ctx = None
    if "world_context" in cfg["scenario"]:
        try:
            world_ctx = WorldContext(cfg["scenario"]["world_context"])
            print(f"[Data Generation] Loaded World Context: {world_ctx.name}")
        except Exception as e:
            print(f"[Data Generation] Warning: Failed to load world context: {e}")

    validator = Validator(
        min_response_tokens=cfg["validation"]["min_response_tokens"],
        max_response_tokens=cfg["validation"]["max_response_tokens"],
    )

    from src.data_gen.turn_generator import TurnGenerator
    from src.data_gen.counterfactual import CounterfactualAugmenter
    from src.data_gen.labeler import Labeler

    n_episodes = cfg["generation"]["n_episodes"]
    scenarios  = bank.sample(n_episodes, weights=scenario_weights)
    seed       = cfg.get("seed", 42)

    # Build one worker (labeler + turn_gen + cf_augmenter + rng) per provider client.
    # Each worker is fully independent — safe to run in parallel threads.
    if args.dry_run:
        raw_clients = [_build_mock_labeler()]
        is_mock = True
    else:
        from src.data_gen.teacher import build_teacher_pool
        raw_clients = build_teacher_pool(cfg)
        is_mock = False

    workers = []
    for w_idx, client in enumerate(raw_clients):
        w_rng = random.Random(seed + w_idx)
        if is_mock:
            w_labeler = client          # MockLabeler is already the labeler
        else:
            prompt_cfg = cfg["prompts"]
            teacher_cfg = (cfg.get("teachers") or [cfg["teacher"]])[w_idx]
            w_labeler = Labeler(
                client=client,
                prompt_config=prompt_cfg,
                model=teacher_cfg.get("model", "Qwen/Qwen3-8B"),
            )
        w_turn_gen = TurnGenerator(
            labeler=w_labeler,
            validator=validator,
            rng=w_rng,
            player_utterance_sampled_ratio=cfg["generation"].get("player_utterance_sampled_ratio", 0.5),
        )
        w_cf = CounterfactualAugmenter(
            labeler=w_labeler,
            validator=validator,
            rng=w_rng,
            n_variants=cfg["generation"].get("counterfactuals_per_episode", 3),
        )
        workers.append((w_turn_gen, w_cf, w_rng))

    # Thread-safe worker checkout queue
    worker_q: queue.Queue = queue.Queue()
    for w in workers:
        worker_q.put(w)

    raw_dir       = Path(cfg["output"]["raw_episodes_dir"])
    validated_dir = Path(cfg["output"]["validated_turns_dir"])
    cf_dir        = Path(cfg["output"]["counterfactuals_dir"])

    # Shared counters protected by a lock
    _lock         = threading.Lock()
    total_turns   = 0
    total_invalid = 0
    total_cf      = 0
    completed     = 0

    n_workers = len(workers)
    print(f"\n[Data Generation] Generating {n_episodes} episodes "
          f"({'DRY-RUN' if args.dry_run else 'REAL'})  "
          f"workers={n_workers}  "
          f"parallel={'yes' if n_workers > 1 else 'no'}...\n")

    def _generate_one(ep_idx: int, scenario: dict) -> None:
        nonlocal total_turns, total_invalid, total_cf, completed

        turn_gen, cf_augmenter, rng = worker_q.get()
        try:
            # Prepare scenario with World Context
            base_world_desc = ""
            npc_def = None

            if world_ctx:
                base_world_desc = (
                    f"{world_ctx.name}: {world_ctx.description}\n"
                    f"Factions: {', '.join(world_ctx.factions.keys())}"
                )
                role_pool = scenario.get("npc_role_pool", list(ROLE_PROFILES.keys()))
                candidate_npcs = [n for n in world_ctx.npcs.values() if n["role"] in role_pool]
                if candidate_npcs and rng.random() < 0.5:
                    npc_def = rng.choice(candidate_npcs)
                    role = npc_def["role"]
                    rels = world_ctx.get_relationships_for_npc(npc_def["npc_id"])
                    if rels:
                        rel_text = "\nRelationships:\n" + "\n".join(
                            f"- {r['type']} with {r['target']}: {r['history']} (Trust={r['stance'].get('trust', '?')})"
                            for r in rels
                        )
                        base_world_desc += rel_text
                else:
                    role = rng.choice([r for r in role_pool if r in ROLE_PROFILES] or list(ROLE_PROFILES.keys()))
            else:
                role_pool = scenario.get("npc_role_pool", list(ROLE_PROFILES.keys()))
                role = rng.choice([r for r in role_pool if r in ROLE_PROFILES] or list(ROLE_PROFILES.keys()))

            scenario = dict(scenario)          # shallow copy — don't mutate shared object
            scenario["world_context"] = base_world_desc

            npc_profile = build_npc_profile(role, scenario, rng, npc_def=npc_def)
            arc         = plan_episode(scenario, npc_profile, rng)
            episode_id  = str(uuid.uuid4())[:8]

            turns = turn_gen.generate_episode(scenario, npc_profile, arc, episode_id=episode_id)

            raw_path = raw_dir / f"ep_{episode_id}.jsonl"
            with open(raw_path, "w") as fh:
                for t in turns:
                    fh.write(json.dumps(t) + "\n")

            valid_turns   = [t for t in turns if "_validation_errors" not in t]
            invalid_count = len(turns) - len(valid_turns)
            ep_cf         = 0

            if valid_turns and not args.dry_run:
                val_path = validated_dir / f"ep_{episode_id}.jsonl"
                with open(val_path, "w") as fh:
                    for t in valid_turns:
                        fh.write(json.dumps(t) + "\n")
                cf_episodes = cf_augmenter.augment(valid_turns, arc)
                for cf_idx, cf_ep in enumerate(cf_episodes):
                    cf_path = cf_dir / f"ep_{episode_id}_cf{cf_idx}.jsonl"
                    with open(cf_path, "w") as fh:
                        for t in cf_ep:
                            fh.write(json.dumps(t) + "\n")
                    ep_cf += len(cf_ep)
            elif valid_turns and args.dry_run:
                cf_episodes = cf_augmenter.augment(valid_turns, arc)
                ep_cf = sum(len(cf) for cf in cf_episodes)

            with _lock:
                total_turns   += len(valid_turns)
                total_invalid += invalid_count
                total_cf      += ep_cf
                completed     += 1
                if completed % max(1, n_episodes // 10) == 0 or completed == n_episodes:
                    print(f"  [{completed}/{n_episodes}] valid={total_turns}  "
                          f"invalid={total_invalid}  cf_turns={total_cf}")
        finally:
            worker_q.put((turn_gen, cf_augmenter, rng))

    max_workers = max(1, len(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_generate_one, idx, scenario)
            for idx, scenario in enumerate(scenarios, 1)
        ]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"[WARNING] Episode failed: {exc}")

    print(f"\n[Generation complete] valid={total_turns}  invalid={total_invalid}  cf={total_cf}\n")


def _run_validation(cfg: dict) -> None:
    from src.data_gen.validator import Validator

    validated_dir = Path(cfg["output"]["validated_turns_dir"])
    validator = Validator(
        min_response_tokens=cfg["validation"]["min_response_tokens"],
        max_response_tokens=cfg["validation"]["max_response_tokens"],
    )

    total = 0
    errors_found = 0
    for fpath in sorted(validated_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total += 1
                errs = []
                errs += validator._check_C_t(record.get("C_t", {}))
                errs += validator._check_A_t(record.get("A_t", {}))
                errs += validator._check_D_t(record.get("D_t", {}), record.get("N_t", {}),
                                              {"allowed_reveal": "full"})
                if errs:
                    errors_found += 1

    print(f"[Validation] total={total}  schema_errors={errors_found}")


def _run_packaging(cfg: dict, args) -> None:
    from src.packaging.packager import Packager

    validated_dir = cfg["output"]["validated_turns_dir"]
    out_dir       = "data/packaged"

    cf_dir = Path(cfg["output"]["counterfactuals_dir"])
    val_dir = Path(validated_dir)

    merged_dir = Path("data/merged_validated")
    merged_dir.mkdir(exist_ok=True)
    for src in list(val_dir.glob("*.jsonl")) + list(cf_dir.glob("*.jsonl")):
        import shutil
        shutil.copy(src, merged_dir / src.name)

    packager = Packager(str(merged_dir), out_dir)
    manifest = packager.build_all()

    if not args.no_mlflow:
        import mlflow
        mlflow.log_dict(manifest, "data_manifest.json")

    print(f"[Packaging] {manifest}")


def _run_split(cfg: dict) -> None:
    from src.packaging.splitter import Splitter

    splitter = Splitter(seed=cfg.get("seed", 42))
    manifest = splitter.split_by_episode(
        packaged_dir="data/packaged",
        splits_dir="data/splits",
    )
    print(f"[Split] {manifest}")


def _build_real_labeler(cfg: dict):
    from src.data_gen.teacher import build_teacher_client
    from src.data_gen.labeler import Labeler

    teacher_cfg = cfg["teacher"]
    provider    = teacher_cfg.get("provider", "local_hf")
    print(f"[Teacher] provider={provider}  model={teacher_cfg.get('model', '')}")

    client = build_teacher_client(teacher_cfg)
    return Labeler(
        client=client,
        prompt_config=cfg["prompts"],
        model=teacher_cfg.get("model", "Qwen/Qwen3-8B"),
    )


def _build_mock_labeler(seed: int = 42):
    return MockLabeler(seed=seed)


class MockLabeler:
    """Deterministic mock labeler for dry-run / CI testing. No API calls."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def label_C_A_M(self, player_utterance, npc_profile, scenario, arc_phase, prior_stance, history):
        """Merged C_t + A_t + M_t labeling (mirrors Labeler.label_C_A_M)."""
        C_t = self.label_C(player_utterance, npc_profile, scenario, arc_phase, history)
        A_t, M_t = self.label_A_M(player_utterance, C_t, npc_profile, scenario, prior_stance, history)
        return C_t, A_t, M_t

    def label_C(self, player_utterance, npc_profile, scenario, arc_phase, history):
        acts = self.rng.choice([["probe"], ["accuse", "probe"], ["ask"], ["threaten"], ["flatter"]])
        return {
            "dialogue_act": acts,
            "tone": self.rng.choice(["neutral", "confrontational", "warm", "evasive"]),
            "risk_type": self.rng.choice(["none", "secret-risk", "face-risk"]),
        }

    def label_A_M(self, player_utterance, C_t, npc_profile, scenario, prior_stance, history):
        A_t = {
            "valence":  self.rng.choice(["negative", "neutral", "positive"]),
            "arousal":  self.rng.choice(["low", "medium", "high"]),
            "threat":   self.rng.choice(["low", "medium", "high"]),
            "control":  self.rng.choice(["low", "medium", "high"]),
        }
        M_t = {
            "player_intent":      self.rng.choice(["seek-info", "trap", "probe", "bond", "test"]),
            "player_knowledge":   self.rng.choice(["unaware", "partial", "informed"]),
            "player_credibility": self.rng.choice(["low", "medium", "high"]),
        }
        return A_t, M_t

    def label_R_N_D(self, player_utterance, C_t, A_t, M_t, npc_profile, scenario,
                    arc, arc_phase, required_shifts, prior_R_t, history, target_policy=""):
        dims = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]
        levels = ["VL", "L", "N", "H", "VH"]
        deltas = ["-", "0", "+"]

        R_t = {}
        for dim in dims:
            prior_level = prior_R_t.get(dim, {})
            if isinstance(prior_level, dict):
                prior_level = prior_level.get("level", "N")
            R_t[dim] = {
                "level": prior_level,
                "delta": self.rng.choice(deltas),
            }

        # Honour required shifts
        for shift in required_shifts:
            var = shift.get("var")
            delta = shift.get("delta", "0")
            if var in dims:
                R_t[var]["delta"] = delta
                idx_map = {"VL": 0, "L": 1, "N": 2, "H": 3, "VH": 4}
                rev_map = {0: "VL", 1: "L", 2: "N", 3: "H", 4: "VH"}
                cur_idx = idx_map.get(R_t[var]["level"], 2)
                if delta in ("+", "++"):
                    R_t[var]["level"] = rev_map.get(min(4, cur_idx + 1), "VH")
                elif delta in ("-", "--"):
                    R_t[var]["level"] = rev_map.get(max(0, cur_idx - 1), "VL")

        allowed_reveal = arc.get("allowed_reveal", "hint")
        reveal_options = {
            "none": ["none"],
            "hint": ["none", "hint"],
            "partial": ["none", "hint", "partial"],
            "full": ["none", "hint", "partial", "full"],
        }.get(allowed_reveal, ["none"])

        N_t = {
            "duty_pressure":    self.rng.choice(["low", "medium", "high"]),
            "secrecy_pressure": self.rng.choice(["low", "medium", "high"]),
            "face_pressure":    self.rng.choice(["low", "medium"]),
            "value_conflict":   self.rng.choice(["none", "mild"]),
        }
        # Use target_policy if provided (simulates policy-targeted generation)
        if target_policy:
            policy = target_policy
        else:
            policy = self.rng.choice(["deflect", "withhold", "challenge", "partial", "soothe",
                                      "answer", "threaten", "negotiate", "test", "clarify"])
        D_t = {
            "response_policy":  policy,
            "reveal_decision":  self.rng.choice(reveal_options),
            "repair_strategy":  self.rng.choice(["none", "soften", "clarify"]),
        }
        return R_t, N_t, D_t

    def generate_response(self, player_utterance, C_t, A_t, M_t, R_t, N_t, D_t,
                          npc_profile, history, **kwargs):
        policy = D_t.get("response_policy", "deflect")
        templates = {
            "deflect":   "That is not something I can discuss here.",
            "withhold":  "I have nothing to say about that matter.",
            "challenge": "You had better watch your words carefully.",
            "partial":   "I may have heard something, but I cannot be certain.",
            "soothe":    "Let us not let this become an argument.",
            "answer":    "Very well, I will tell you what I know.",
            "threaten":  "Push further and you will regret it.",
            "negotiate": "Perhaps we can reach an arrangement.",
            "test":      "Why do you want to know that?",
        }
        base = templates.get(policy, "I have no comment on that.")
        return base

    def generate_scene_opening(self, scenario, npc_profile):
        setting  = scenario.get("setting", "unknown place")
        role     = npc_profile.get("role", "guard")
        s_type   = scenario.get("scenario_type", "")
        return {
            "scene_description": (
                f"You stand in the {setting}. The {role} regards you with cautious eyes. "
                f"The air is thick with unspoken tension."
            ),
            "player_opening": {
                "secret_extraction":    "I need to ask you about something important.",
                "apology_repair":       "I owe you an apology for what happened.",
                "alliance_negotiation": "I have a proposal that could benefit us both.",
                "rumor_confrontation":  "I've been hearing things about you.",
                "threat_escalation":    "Stand aside — I need to pass.",
                "trust_building":       "I was hoping we could talk.",
                "deception_detection":  "I need to verify your story one more time.",
            }.get(s_type, "We need to talk."),
        }

    def generate_player_utterance(self, scenario, arc_phase, npc_profile, history):
        return self.rng.choice([
            "I see what you mean, but I need more details.",
            "That sounds reasonable. Go on.",
            "I'm not sure I believe that.",
            "Can you explain why you did that?",
            "Look, we don't have much time."
        ])


if __name__ == "__main__":
    main()
