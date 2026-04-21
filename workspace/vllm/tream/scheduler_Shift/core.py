from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .anchor import AnchorTracker
from .candidates import CandidateGenerator
from .config_space import ConfigSpace
from .hold_policy import HoldAnchorPolicy
from .logging import build_phase_log_meta
from .pareto import AnchorManager, pareto_filter
from .phases import PhaseMachine, PhaseSignals
from .phases import Phase
from .policies.base import SchedulerContext
from .regime import RegimeQuantizer, ShockDetector
from .reroute_policy import LocalReroutePolicy, ReroutePolicyConfig
from .safety import SafetyFilter, SafetyResult
from .salvage_policy import SalvagePolicy
from .seek_policy import SeekAnchorPolicy, SeekPolicyConfig
from .stats import StatsStore
from .transitions import max_fast_step_for_phase
from .types import ConfigX, Decision, DerivedMetrics, WindowMetrics


@dataclass
class SchedulerConfig:
    # Safety
    tau: Optional[float] = None
    sla: Optional[float] = None
    mem_limit: Optional[float] = None
    train_min: Optional[float] = None
    beta: float = 1.0
    sigma_floor: float = 1e-3
    min_count_for_trust: int = 2
    max_staleness_ticks: int = 64
    safety_latency_metric: str = "lat_mean"  # MVP: lat_mean, v2: lat_p95.

    # Shock + regime
    shock_w_drift: float = 1.0
    shock_w_accept: float = 1.0
    shock_w_reverify: float = 1.0
    shock_w_verify_ratio: float = 1.0
    shock_w_waste_rate: float = 1.0
    shock_f_drift: float = 0.02
    shock_f_accept: float = 0.05
    shock_f_reverify: float = 500.0
    shock_f_verify_ratio: float = 0.05
    shock_f_waste_rate: float = 0.05
    shock_component_cap: float = 1.0
    shock_t_high: float = 0.35
    shock_t_low: float = 0.20
    adapt_hold: int = 2
    shock_use_jsd_only: bool = False
    drift_bins: Sequence[float] = (0.02, 0.05, 0.10)
    accept_bins: Sequence[float] = (0.60, 0.80, 0.92)
    verify_ratio_bins: Sequence[float] = (0.30, 0.55, 0.75)

    # Candidate/anchors
    candidate_cap: int = 64
    anchor_k: int = 8
    stable_eps_fast: int = 1
    stable_eps_slow: int = 0
    adapt_eps_fast: int = 2
    adapt_eps_slow: int = 1
    probe_deltas: Dict[str, int] = field(
        default_factory=lambda: {"ctx": 1, "inf": 1, "cross": True}
    )
    candidate_min_probe_keep: int = 2
    adapt_use_all_configs_if_small: bool = True

    # Policy
    preference: str = "quality"
    switch_cost_lambda: float = 0.0
    dwell_steps: int = 2

    # Stats
    stats_alpha: float = 0.2
    history_size: int = 128

    # Cold-start exploration
    warmup_hold_windows: int = 4
    warmup_explore_enable: bool = False
    warmup_force_descend_large: bool = False
    warmup_large_ctx_threshold: int = 6
    warmup_large_inf_threshold: int = 6
    warmup_oscillation_exit_enable: bool = False
    warmup_oscillation_patience: int = 4
    warmup_oscillation_unique_max: int = 2
    cold_start_windows: int = 8  # minimum cold-start windows
    cold_start_max_windows: int = 12
    cold_start_post_pause_windows: int = 1  # hold after cold-start exits
    cold_start_probe_every: int = 2
    cold_start_relax_safety: bool = True
    cold_start_axis_rotation: bool = True
    cold_start_i_major_span: int = 3  # i-i-i-c pattern by default
    cold_start_probe_eps_fast: int = 2
    cold_start_probe_eps_slow: int = 0
    cold_start_avoid_two_cycle: bool = True
    cold_start_directional_enable: bool = False
    cold_start_prefer_large_descend: bool = False
    cold_start_prefer_small_ascend: bool = False
    cold_start_large_ctx_threshold: int = 6
    cold_start_large_inf_threshold: int = 6
    cold_start_small_ctx_threshold: int = 1
    cold_start_small_inf_threshold: int = 1
    cold_start_target_ctx: int = 2
    cold_start_target_inf: int = 2
    cold_start_target_ctx_band: int = 1
    cold_start_target_inf_band: int = 1
    cold_start_single_axis_lock: bool = False
    cold_start_early_exit_enable: bool = False
    cold_start_target_stable_windows: int = 2
    cold_start_relax_radius: int = 1
    cold_start_relax_quality_slack: float = 0.0
    cold_start_relax_latency_slack: float = 0.0
    cold_start_patience_directions: int = 3  # stop after N probe directions without improvement
    cold_start_improve_latency_eps: float = 0.0
    cold_start_improve_quality_eps: float = 0.0
    seek_exit_anchor_confidence: float = 0.60
    seek_exit_min_probe_observations: int = 3
    seek_exit_quality_margin: float = -0.02
    seek_exit_latency_margin: float = -0.05
    # Deadlock breaker under strict safety: local whitelist probing only.
    cold_whitelist_probe_enable: bool = True
    cold_whitelist_probe_budget: int = 4
    cold_whitelist_latency_slack: float = 0.15
    cold_whitelist_quality_slack: float = 0.6
    cold_whitelist_max_switch_distance: int = 2
    # ADAPT probe->commit
    adapt_probe_windows: int = 2
    adapt_exit_quality_margin: float = 0.03
    adapt_exit_latency_margin: float = 0.08
    seek_exit_stable_windows: int = 0
    seek_exit_stable_radius: int = 0
    salvage_infeasible_streak: int = 2
    salvage_dwell_steps: int = 4
    min_switch_gain_eps: float = 0.01
    # Optional post-policy stabilizer (default OFF for zero behavior change).
    stabilizer_enable: bool = False
    stabilizer_shadow: bool = True
    stabilizer_allow_in_adapt: bool = False
    stabilizer_two_cycle: bool = True
    stabilizer_seek_monotone: bool = False
    stabilizer_seek_max_inf_rebound: int = 0
    stabilizer_low_shock_max: float = 0.20
    stabilizer_low_shock_gain_eps: float = 0.0
    force_adapt_min_count: int = 3
    # Near-violation trigger: force ADAPT even if shock is low.
    force_adapt_quality_margin: float = 0.02
    force_adapt_latency_margin: float = 0.05
    anchor_bootstrap_windows: int = 3
    anchor_init_confidence: float = 0.15
    anchor_confidence_gain: float = 0.25
    anchor_confidence_decay_hold_drift: float = 0.05
    anchor_trial_promote_windows: int = 2
    hold_anchor_radius: int = 1
    reroute_radius: int = 2
    reroute_guard_min_attempts: int = 6
    reroute_guard_min_success_rate: float = 0.20
    reroute_guard_fail_streak: int = 4
    reroute_guard_cooldown_windows: int = 3
    local_relax_enable: bool = True
    local_relax_windows: int = 3
    local_relax_radius: int = 1
    local_relax_drift_threshold: float = 0.30
    local_relax_quality_slack: float = 0.05
    local_relax_latency_slack: float = 0.08
    # JSD-triggered local probe session in SHOCK_LOCAL_REROUTE:
    # probe_down -> probe_up -> commit_best.
    jsd_local_probe_enable: bool = True
    jsd_local_probe_up_max_step: int = 1
    jsd_local_probe_down_max_step: int = 2
    jsd_local_probe_down_step_mild: int = 1
    jsd_local_probe_down_step_severe: int = 2
    # JSD refresh + axis-priority local search behavior.
    jsd_probe_force_refresh: bool = True
    jsd_probe_invalidate_radius: int = 3
    jsd_probe_invalidate_hold_ticks: int = 256
    jsd_probe_axis_order: str = "ctx,inf"
    jsd_probe_continue_steps: int = 1
    jsd_probe_state_hold_windows: int = 3


class SchedulerCore:
    """Composable scheduler core: metrics -> mode/regime -> safe candidates -> policy decision."""

    def __init__(
        self,
        *,
        config_space: ConfigSpace,
        initial_config: ConfigX,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.cfg = scheduler_config or SchedulerConfig()
        self.config_space = config_space
        self.x_prev = initial_config
        self.x_init = initial_config
        self.mode = "STABLE"
        self.regime_id = "d-1_a-1_v-1"
        self.last_decision: Optional[Decision] = None
        self.anchors_by_regime: Dict[str, List[ConfigX]] = {}
        self.history: List[Mapping[str, object]] = []
        self.x_prev_prev: Optional[ConfigX] = None
        self.dwell_state: Dict[str, int] = {
            "remaining": 0,
            "infeasible_streak": 0,
            "degradation_streak": 0,
            "adapt_probe_used": 0,
        }
        self.x_safe_default = initial_config
        self.total_steps = 0
        self.adapt_probe_remaining = 0
        self.adapt_windows_elapsed = 0
        self.cold_probe_count = 0
        self.cold_whitelist_probe_remaining = max(0, int(self.cfg.cold_whitelist_probe_budget))
        self.cold_start_no_improve_streak = 0
        self.cold_start_probe_observations = 0
        self.cold_start_best_latency: Optional[float] = None
        self.cold_start_best_quality: Optional[float] = None
        self.cold_start_terminated = False
        self.prev_cold_start_active = True
        self.post_cold_pause_remaining = 0
        self.seek_anchor_stable_streak = 0
        self.local_relax_remaining = 0
        self.jsd_probe_state: Dict[str, Any] = {}
        self.reroute_guard_state: Dict[str, float] = {
            "attempts": 0.0,
            "success": 0.0,
            "fail_streak": 0.0,
            "cooldown": 0.0,
        }
        self.phase_machine = PhaseMachine()
        self.anchor_tracker = AnchorTracker()

        self.shock_detector = ShockDetector(
            w_drift=self.cfg.shock_w_drift,
            w_accept=self.cfg.shock_w_accept,
            w_reverify=self.cfg.shock_w_reverify,
            w_verify_ratio=self.cfg.shock_w_verify_ratio,
            w_waste_rate=self.cfg.shock_w_waste_rate,
            f_drift=self.cfg.shock_f_drift,
            f_accept=self.cfg.shock_f_accept,
            f_reverify=self.cfg.shock_f_reverify,
            f_verify_ratio=self.cfg.shock_f_verify_ratio,
            f_waste_rate=self.cfg.shock_f_waste_rate,
            component_cap=self.cfg.shock_component_cap,
            t_high=self.cfg.shock_t_high,
            t_low=self.cfg.shock_t_low,
            adapt_hold=self.cfg.adapt_hold,
        )
        self.regime_quantizer = RegimeQuantizer(
            drift_bins=self.cfg.drift_bins,
            accept_bins=self.cfg.accept_bins,
            verify_ratio_bins=self.cfg.verify_ratio_bins,
        )
        self.stats_store = StatsStore(default_alpha=self.cfg.stats_alpha)
        self.candidate_gen = CandidateGenerator(
            config_space,
            cap_size=self.cfg.candidate_cap,
            default_eps_fast=self.cfg.stable_eps_fast,
            default_eps_slow=self.cfg.stable_eps_slow,
        )
        self.safety_filter = SafetyFilter(
            tau=self.cfg.tau,
            sla=self.cfg.sla,
            mem_limit=self.cfg.mem_limit,
            train_min=self.cfg.train_min,
            beta=self.cfg.beta,
            sigma_floor=self.cfg.sigma_floor,
            min_count_for_trust=self.cfg.min_count_for_trust,
            max_staleness_ticks=self.cfg.max_staleness_ticks,
            latency_metric=self.cfg.safety_latency_metric,
        )
        self.anchor_manager = AnchorManager()
        self.seek_policy = SeekAnchorPolicy(
            SeekPolicyConfig(
                probe_eps_fast=max(1, int(self.cfg.cold_start_probe_eps_fast)),
                probe_eps_slow=max(0, int(self.cfg.cold_start_probe_eps_slow)),
                avoid_two_cycle=bool(self.cfg.cold_start_avoid_two_cycle),
                quality_floor_delta=0.03,
                whitelist_latency_slack=max(0.0, float(self.cfg.cold_whitelist_latency_slack)),
                whitelist_quality_slack=max(0.0, float(self.cfg.cold_whitelist_quality_slack)),
                whitelist_max_switch_distance=max(
                    0, int(self.cfg.cold_whitelist_max_switch_distance)
                ),
            )
        )
        self.hold_policy = HoldAnchorPolicy()
        self.reroute_policy = LocalReroutePolicy(
            ReroutePolicyConfig(reroute_radius=max(1, int(self.cfg.reroute_radius)))
        )
        self.salvage_policy = SalvagePolicy()

    def step(self, window_metrics: WindowMetrics) -> Decision:
        # 1) Derive helper metrics (verify_ratio / waste_rate / margins).
        derived = self._derive_metrics(window_metrics)
        adapt_exit_ready = self._adapt_exit_ready(derived)
        warmup_active = self.total_steps < int(max(0, self.cfg.warmup_hold_windows))
        prev_cold_start_active = bool(self.prev_cold_start_active)
        # Update cold-start progress from previous probe observation before deciding
        # whether this window still belongs to cold-start.
        self._update_cold_start_progress(window_metrics)
        self._update_jsd_probe_observation(window_metrics)
        cold_start_active = self._compute_cold_start_active()
        if prev_cold_start_active and not cold_start_active:
            self.post_cold_pause_remaining = max(
                int(self.post_cold_pause_remaining),
                max(0, int(self.cfg.cold_start_post_pause_windows)),
            )
        cold_start_pause_active = int(self.post_cold_pause_remaining) > 0

        # 2) Shock detect -> mode.
        prev_mode = self.mode
        shock = self.shock_detector.update(derived, can_exit_adapt=adapt_exit_ready)
        self.mode = shock.mode
        if warmup_active:
            # Warm-up windows should collect statistics without entering ADAPT.
            self.mode = "STABLE"
            self.shock_detector.force_stable()

        # 3) Regime quantize.
        self.regime_id = self.regime_quantizer.quantize(
            drift=window_metrics.drift_value(),
            accept=window_metrics.spec_accept_mean,
            verify_ratio=derived.verify_ratio,
        )

        # 4) Update recency stats for previous config.
        self.stats_store.update(
            self.regime_id,
            self.x_prev,
            self._stats_payload(window_metrics, derived),
            alpha=self.cfg.stats_alpha,
        )
        _, current_sample_count = self.stats_store.get_last_seen_count(
            self.regime_id, self.x_prev
        )
        force_adapt_reasons = self._force_adapt_reasons(
            derived,
            sample_count=current_sample_count,
            warmup_active=warmup_active,
        )
        if force_adapt_reasons:
            self.mode = "ADAPT"
        if prev_mode != "ADAPT" and self.mode == "ADAPT":
            self.adapt_probe_remaining = max(0, int(self.cfg.adapt_probe_windows))
            self.adapt_windows_elapsed = 0
        elif self.mode != "ADAPT":
            self.adapt_probe_remaining = 0
            self.adapt_windows_elapsed = 0
        seek_exit_ready = self._seek_exit_ready(derived)
        if cold_start_active and seek_exit_ready:
            cold_start_active = False
            self.cold_start_terminated = True
        local_relax_active = self._update_local_relax_state(
            mode=self.mode,
            warmup_active=warmup_active,
            cold_start_active=(cold_start_active or cold_start_pause_active),
            drift_signal=derived.shock_drift,
        )
        reroute_guard_hold = self._reroute_guard_hold_this_step() or cold_start_pause_active

        phase_transition = self.phase_machine.update(
            PhaseSignals(
                step=self.total_steps,
                warmup_active=warmup_active,
                cold_start_active=cold_start_active,
                seek_exit_ready=seek_exit_ready,
                mode=self.mode,
                salvage_active=False,
                reroute_guard_hold=reroute_guard_hold,
            )
        )

        # 5) Candidate generation.
        anchors = self.anchors_by_regime.get(self.regime_id, [])
        candidates = self.candidate_gen.generate(
            x_prev=self.x_prev,
            anchors=anchors,
            mode=self.mode,
            eps_fast=self.cfg.adapt_eps_fast if self.mode == "ADAPT" else self.cfg.stable_eps_fast,
            eps_slow=self.cfg.adapt_eps_slow if self.mode == "ADAPT" else self.cfg.stable_eps_slow,
            probe_deltas=self.cfg.probe_deltas,
            safe_default=self.x_safe_default,
            min_probe_keep=self.cfg.candidate_min_probe_keep if self.mode == "ADAPT" else 0,
        )
        if self.mode == "ADAPT" and bool(self.cfg.adapt_use_all_configs_if_small):
            try:
                all_candidates = self.config_space.all_configs()
            except ValueError:
                all_candidates = None
            if all_candidates is not None and (
                self.cfg.candidate_cap <= 0 or len(all_candidates) <= int(self.cfg.candidate_cap)
            ):
                candidates = list(all_candidates)

        # 6) Safety filter.
        if cold_start_active and self.cfg.cold_start_relax_safety:
            safe_set = list(candidates) if candidates else [self.x_prev]
            safe = SafetyResult(safe_set=safe_set, margin_report={}, salvaged=False)
        else:
            safe = self.safety_filter.filter_candidates(
                regime_id=self.regime_id,
                candidates=candidates,
                stats_store=self.stats_store,
                x_safe_default=self.x_safe_default,
                x_prev=self.x_prev,
            )
            safe_set = safe.safe_set or [self.x_prev]
            if local_relax_active:
                safe_set = self._augment_safe_set_local_relax(
                    safe_set=safe_set,
                    candidates=candidates,
                    margin_report=safe.margin_report,
                )

        # 7) Pareto + anchors update (P_hat computed on trusted-recent points only).
        trusted_safe_set, trust_meta = self._trusted_subset(safe_set)
        frontier_pool = list(trusted_safe_set)
        trust_fallback = False
        if not frontier_pool:
            trust_fallback = True
            if self._contains_config(safe_set, self.x_prev):
                frontier_pool = [self.x_prev]
            elif safe_set:
                frontier_pool = [safe_set[0]]
            else:
                frontier_pool = [self.x_prev]

        mu_vectors = self._mu_vectors(frontier_pool, window_metrics)
        pareto_set = self._pareto_configs(frontier_pool, mu_vectors)
        anchors = self.anchor_manager.update_anchors(
            frontier_pool,
            mu_vectors,
            self.cfg.anchor_k,
            regime_id=self.regime_id,
        )
        if not anchors:
            anchors = list(frontier_pool)[: max(1, int(self.cfg.anchor_k))]
        self.anchors_by_regime[self.regime_id] = anchors

        # 8) Policy select_next.
        adapt_probe_enabled = self.mode == "ADAPT" and self.adapt_probe_remaining > 0

        decision: Optional[Decision] = None
        reroute_attempted = False

        if warmup_active:
            warmup_target = (
                self.x_safe_default
                if self._contains_config(safe_set, self.x_safe_default)
                else self.x_prev
            )
            if not self._contains_config(safe_set, warmup_target) and safe_set:
                warmup_target = safe_set[0]
            self.adapt_probe_remaining = 0
            adapt_probe_enabled = False
            decision = Decision(
                x_next=warmup_target,
                reason="warmup_hold_safe_default",
                mode="STABLE",
                regime_id=self.regime_id,
                meta={
                    "warmup_active": True,
                    "warmup_remaining": float(
                        max(0, int(self.cfg.warmup_hold_windows) - int(self.total_steps) - 1)
                    ),
                },
            )

        if decision is None and cold_start_pause_active:
            decision = Decision(
                x_next=self.x_prev,
                reason="post_cold_start_pause_hold",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={
                    "cold_start_pause_active": True,
                    "cold_start_pause_remaining": float(self.post_cold_pause_remaining),
                },
            )

        active_phase = self.phase_machine.state.phase
        anchor_cfg = self.anchor_tracker.state.anchor_cfg
        force_salvage = self._should_force_quality_salvage(
            derived,
            safe,
            sample_count=current_sample_count,
            warmup_active=warmup_active,
        )
        if decision is None and force_salvage:
            transition = self.phase_machine.update(
                PhaseSignals(
                    step=self.total_steps,
                    warmup_active=warmup_active,
                    cold_start_active=cold_start_active,
                    seek_exit_ready=seek_exit_ready,
                    mode=self.mode,
                    salvage_active=True,
                    reroute_guard_hold=reroute_guard_hold,
                )
            )
            if transition is not None:
                phase_transition = transition
            active_phase = Phase.SALVAGE

        if decision is None and active_phase == Phase.SALVAGE:
            adapt_probe_enabled = False
            decision = self.salvage_policy.select_next(
                mode=self.mode,
                regime_id=self.regime_id,
                x_prev=self.x_prev,
                x_safe_default=self.x_safe_default,
                safe_set=safe_set,
                margin_report=safe.margin_report,
                derived=derived,
                config_space=self.config_space,
                stable_eps_fast=self.cfg.stable_eps_fast,
                stable_eps_slow=self.cfg.stable_eps_slow,
            )
            decision.meta["infeasible_streak"] = float(self.dwell_state.get("infeasible_streak", 0))
            decision.meta["force_dwell_steps"] = float(max(1, int(self.cfg.salvage_dwell_steps)))

        if decision is None and active_phase == Phase.SEEK_ANCHOR:
            use_seek = (
                cold_start_active
                and int(max(1, self.cfg.cold_start_probe_every)) > 0
                and (self.total_steps % int(max(1, self.cfg.cold_start_probe_every)) == 0)
            )
            if use_seek:
                seek_decision = self.seek_policy.select_next(
                    mode=self.mode,
                    regime_id=self.regime_id,
                    x_prev=self.x_prev,
                    x_prev_prev=self.x_prev_prev,
                    safe_set=safe_set,
                    candidates=candidates,
                    margin_report=safe.margin_report,
                    window=window_metrics,
                    config_space=self.config_space,
                    stats_store=self.stats_store,
                    tau=self.cfg.tau,
                    preferred_axis=self._cold_start_preferred_axis(),
                    force_adapt_reasons=force_adapt_reasons,
                    whitelist_probe_enable=bool(self.cfg.cold_whitelist_probe_enable),
                    whitelist_probe_remaining=int(self.cold_whitelist_probe_remaining),
                    cold_start_relax_safety=bool(self.cfg.cold_start_relax_safety),
                )
                if seek_decision is not None:
                    decision = seek_decision
                    self.cold_probe_count += 1
                    if decision.reason == "seek_anchor_whitelist_break":
                        decision.meta["whitelist_budget_before"] = float(self.cold_whitelist_probe_remaining)

        if decision is None and active_phase == Phase.SHOCK_LOCAL_REROUTE:
            reroute_attempted = True
            if not self._probe_abort_for_reroute(derived):
                decision = self._maybe_jsd_probe_decision(
                    safe_set=safe_set,
                    candidates=candidates,
                    margin_report=safe.margin_report,
                    window=window_metrics,
                    derived=derived,
                    local_relax_active=local_relax_active,
                )
                if decision is None:
                    decision = self.reroute_policy.select_next(
                        mode=self.mode,
                        regime_id=self.regime_id,
                        x_prev=self.x_prev,
                        anchor_cfg=anchor_cfg,
                        safe_set=safe_set,
                        margin_report=safe.margin_report,
                        config_space=self.config_space,
                        stats_store=self.stats_store,
                        tau=self.cfg.tau,
                        force_adapt_reasons=force_adapt_reasons,
                        accept_delta=float(shock.components.get("accept", 0.0)),
                        allow_local_uphill=local_relax_active,
                    )
                if decision is not None:
                    decision.meta["adapt_probe_window"] = 1.0 if adapt_probe_enabled else 0.0
                    decision.meta["adapt_probe_remaining"] = float(self.adapt_probe_remaining)
                    decision.meta["accept_delta"] = float(shock.components.get("accept", 0.0))
                    decision.meta["shock_score"] = float(shock.shock_score)

        if decision is None and active_phase == Phase.HOLD_ANCHOR:
            hold_safe_set = self._anchor_protected_pool(
                safe_set=safe_set,
                anchor_cfg=anchor_cfg,
                radius=max(1, int(self.cfg.hold_anchor_radius)),
            )
            hold_ctx = SchedulerContext(
                mode=self.mode,
                regime_id=self.regime_id,
                x_prev=self.x_prev,
                safe_set=hold_safe_set,
                pareto_set=pareto_set,
                anchors=anchors,
                stats_store=self.stats_store,
                config_space=self.config_space,
                preference=self.cfg.preference,
                switch_cost_lambda=self.cfg.switch_cost_lambda,
                dwell_state=self.dwell_state,
                derived=derived,
                history=list(self.history),
                alarm_severity=self._alarm_severity(derived, shock.shock_score),
                allow_uncertainty_exploration=False,
                stable_eps_fast=self.cfg.stable_eps_fast,
                stable_eps_slow=self.cfg.stable_eps_slow,
                adapt_eps_fast=self.cfg.adapt_eps_fast,
                adapt_eps_slow=self.cfg.adapt_eps_slow,
            )
            decision = self.hold_policy.select_next(
                ctx=hold_ctx,
                anchor_cfg=anchor_cfg,
                anchor_protected_pool=hold_safe_set,
            )

        if decision is None:
            decision = self._phase_fallback_decision(
                phase=active_phase,
                safe_set=safe_set,
                anchor_cfg=anchor_cfg,
                allow_local_uphill=local_relax_active,
            )

        decision = self._apply_switch_guards(
            decision=decision,
            safe_set=safe_set,
            window=window_metrics,
            derived=derived,
            phase=active_phase,
            shock_score=float(shock.shock_score),
            force_adapt_reasons=force_adapt_reasons,
        )
        self._update_reroute_guard(
            phase=active_phase,
            decision=decision,
            attempted=reroute_attempted,
        )
        salvage_active = "salvage" in str(decision.reason).lower()
        if salvage_active:
            transition = self.phase_machine.update(
                PhaseSignals(
                    step=self.total_steps,
                    warmup_active=warmup_active,
                    cold_start_active=cold_start_active,
                    seek_exit_ready=seek_exit_ready,
                    mode=self.mode,
                    salvage_active=True,
                    reroute_guard_hold=reroute_guard_hold,
                )
            )
            if transition is not None:
                phase_transition = transition
        decision.mode = decision.mode or self.mode
        decision.regime_id = decision.regime_id or self.regime_id
        decision.meta = dict(decision.meta or {})

        report = safe.margin_report.get(decision.x_next.hash_key(), {})
        for key in ("quality_margin", "latency_margin", "mem_margin", "train_tps_margin"):
            if key in report:
                decision.safety_margins[key] = float(report[key])
        if safe.salvaged:
            decision.meta["violation_recovery"] = True
            if safe.salvage_choice is not None:
                decision.meta["violation_recovery_choice"] = safe.salvage_choice.to_dict()
            if safe.salvage_reason:
                decision.meta["violation_recovery_reason"] = safe.salvage_reason

        decision.meta["shock_score"] = float(shock.shock_score)
        decision.meta["shock_mode"] = shock.mode
        decision.meta["accept_delta"] = float(shock.components.get("accept", 0.0))
        decision.meta["trigger_meta"] = {
            "shock_score": float(shock.shock_score),
            "mode": str(shock.mode),
            "accept_delta": float(shock.components.get("accept", 0.0)),
            "drift_signal": float(shock.components.get("drift", 0.0)),
        }
        decision.meta["warmup_active"] = bool(warmup_active)
        decision.meta["warmup_hold_windows"] = float(max(0, int(self.cfg.warmup_hold_windows)))
        decision.meta["sample_count_prev_config"] = float(max(0, int(current_sample_count)))
        decision.meta["cold_start_active"] = bool(cold_start_active)
        decision.meta["cold_start_pause_active"] = bool(cold_start_pause_active)
        decision.meta["cold_start_post_pause_windows"] = float(
            max(0, int(self.cfg.cold_start_post_pause_windows))
        )
        decision.meta["cold_start_post_pause_remaining"] = float(
            max(0, int(self.post_cold_pause_remaining))
        )
        decision.meta["cold_start_probe_observations"] = float(self.cold_start_probe_observations)
        decision.meta["cold_start_no_improve_streak"] = float(self.cold_start_no_improve_streak)
        decision.meta["cold_start_terminated"] = 1.0 if self.cold_start_terminated else 0.0
        decision.meta["seek_anchor_stable_streak"] = float(self.seek_anchor_stable_streak)
        decision.meta["seek_exit_ready"] = 1.0 if seek_exit_ready else 0.0
        if self.cold_start_best_latency is not None:
            decision.meta["cold_start_best_latency"] = float(self.cold_start_best_latency)
        if self.cold_start_best_quality is not None:
            decision.meta["cold_start_best_quality"] = float(self.cold_start_best_quality)
        decision.meta["force_adapt"] = bool(force_adapt_reasons)
        if force_adapt_reasons:
            decision.meta["force_adapt_reasons"] = list(force_adapt_reasons)
        decision.meta["adapt_exit_ready"] = 1.0 if adapt_exit_ready else 0.0
        decision.meta["adapt_hold_counter"] = float(shock.adapt_hold_counter)

        adapt_probe_remaining_before = int(self.adapt_probe_remaining)
        if self.mode == "ADAPT":
            if self.adapt_probe_remaining > 0:
                self.adapt_probe_remaining -= 1
            self.adapt_windows_elapsed += 1
        else:
            self.adapt_windows_elapsed = 0
        decision.meta["adapt_probe_remaining_before"] = float(adapt_probe_remaining_before)
        decision.meta["adapt_probe_remaining_after"] = float(self.adapt_probe_remaining)
        decision.meta["adapt_probe_remaining"] = float(self.adapt_probe_remaining)
        decision.meta["adapt_windows_elapsed"] = float(self.adapt_windows_elapsed)
        decision.meta["adapt_probe_budget_exhausted"] = (
            1.0 if self.mode == "ADAPT" and self.adapt_probe_remaining <= 0 else 0.0
        )

        def _descending_count(ref_cfg: ConfigX, pool: Sequence[ConfigX]) -> int:
            ref_ctx = int(ref_cfg.ctx)
            ref_inf = int(ref_cfg.inf)
            ref_key = ref_cfg.hash_key()
            cnt = 0
            for cand in pool:
                if cand.hash_key() == ref_key:
                    continue
                c_ctx = int(cand.ctx)
                c_inf = int(cand.inf)
                if c_ctx <= ref_ctx and c_inf <= ref_inf and (c_ctx < ref_ctx or c_inf < ref_inf):
                    cnt += 1
            return cnt

        desc_prev_safe = _descending_count(self.x_prev, safe_set)
        desc_prev_trusted = _descending_count(self.x_prev, trusted_safe_set)
        desc_init_safe = _descending_count(self.x_init, safe_set)
        desc_init_trusted = _descending_count(self.x_init, trusted_safe_set)

        decision.meta["candidate_size"] = float(len(candidates))
        decision.meta["safe_set_size"] = float(len(safe_set))
        decision.meta["trusted_safe_size"] = float(len(trusted_safe_set))
        decision.meta["descending_from_prev_safe_count"] = float(desc_prev_safe)
        decision.meta["descending_from_prev_trusted_count"] = float(desc_prev_trusted)
        decision.meta["descending_from_init_safe_count"] = float(desc_init_safe)
        decision.meta["descending_from_init_trusted_count"] = float(desc_init_trusted)
        decision.meta["frontier_pool_size"] = float(len(frontier_pool))
        decision.meta["trusted_safe_fallback"] = 1.0 if trust_fallback else 0.0
        decision.meta["p_hat_safe_size"] = float(len(pareto_set))
        decision.meta["anchors_size"] = float(len(anchors))
        decision.meta["p_hat_safe"] = [self._compact_cfg(cfg) for cfg in pareto_set]
        decision.meta["anchors"] = [self._compact_cfg(cfg) for cfg in anchors]
        decision.meta["trust_gate"] = {
            "min_count_for_trust": float(self.cfg.min_count_for_trust),
            "max_staleness_ticks": -1.0
            if self.cfg.max_staleness_ticks is None
            else float(self.cfg.max_staleness_ticks),
            "tracked_safe_size": float(len(trust_meta)),
        }
        whitelist_before = int(self.cold_whitelist_probe_remaining)
        if decision.reason == "seek_anchor_whitelist_break" and self.cold_whitelist_probe_remaining > 0:
            self.cold_whitelist_probe_remaining -= 1
        decision.meta["cold_whitelist_probe_enable"] = bool(self.cfg.cold_whitelist_probe_enable)
        decision.meta["cold_whitelist_probe_budget"] = float(
            max(0, int(self.cfg.cold_whitelist_probe_budget))
        )
        decision.meta["cold_whitelist_probe_remaining_before"] = float(whitelist_before)
        decision.meta["cold_whitelist_probe_remaining_after"] = float(
            self.cold_whitelist_probe_remaining
        )
        untrusted = [
            key
            for key, row in trust_meta.items()
            if float(row.get("is_trusted", 0.0)) <= 0.0
        ]
        if untrusted:
            decision.meta["untrusted_safe_keys"] = untrusted[:16]
        decision.meta["guard_meta"] = {
            "dwell_remaining": float(self.dwell_state.get("remaining", 0)),
            "min_switch_gain_eps": float(self.cfg.min_switch_gain_eps),
            "hold_anchor_radius": float(max(1, int(self.cfg.hold_anchor_radius))),
            "reroute_radius": float(max(1, int(self.cfg.reroute_radius))),
        }
        decision.meta["config_cold_start_relax_safety"] = bool(self.cfg.cold_start_relax_safety)
        decision.meta["config_min_count_for_trust"] = float(self.cfg.min_count_for_trust)
        decision.meta["config_max_staleness_ticks"] = -1.0 if self.cfg.max_staleness_ticks is None else float(self.cfg.max_staleness_ticks)
        decision.meta["local_relax_enable"] = bool(self.cfg.local_relax_enable)
        decision.meta["local_relax_active"] = 1.0 if local_relax_active else 0.0
        decision.meta["local_relax_remaining"] = float(self.local_relax_remaining)
        decision.meta["local_relax_radius"] = float(max(1, int(self.cfg.local_relax_radius)))
        decision.meta["local_relax_drift_threshold"] = float(max(0.0, self.cfg.local_relax_drift_threshold))
        decision.meta["local_relax_quality_slack"] = float(max(0.0, self.cfg.local_relax_quality_slack))
        decision.meta["local_relax_latency_slack"] = float(max(0.0, self.cfg.local_relax_latency_slack))
        decision.meta["reroute_guard_hold"] = 1.0 if reroute_guard_hold else 0.0
        decision.meta["reroute_guard"] = self._reroute_guard_meta()

        phase_name = self.phase_machine.state.phase.value
        if phase_name == "WARMUP":
            candidate_pool_type = "warmup_safe_set"
        elif phase_name == "SEEK_ANCHOR":
            candidate_pool_type = "seek_safe_set"
        elif phase_name == "SHOCK_LOCAL_REROUTE":
            candidate_pool_type = "reroute_safe_set"
        elif phase_name == "SALVAGE":
            candidate_pool_type = "salvage_safe_set"
        else:
            candidate_pool_type = "hold_safe_set"

        trial_promoted = self.anchor_tracker.update(
            phase=self.phase_machine.state.phase,
            regime_id=self.regime_id,
            decision=decision,
            derived=derived,
            step=self.total_steps,
            promote_windows=self.cfg.anchor_trial_promote_windows,
            init_promote_windows=self.cfg.anchor_bootstrap_windows,
            init_confidence=self.cfg.anchor_init_confidence,
            hold_confidence_gain=self.cfg.anchor_confidence_gain,
            hold_confidence_decay=self.cfg.anchor_confidence_decay_hold_drift,
        )
        decision.meta.update(
            build_phase_log_meta(
                phase_state=self.phase_machine.state,
                transition=phase_transition,
                anchor_state=self.anchor_tracker.state,
                candidate_pool_type=candidate_pool_type,
                switch_block_reason=(
                    None
                    if "blocked_reason" not in decision.meta
                    else str(decision.meta.get("blocked_reason"))
                ),
                trial_promoted=trial_promoted,
            )
        )
        decision.meta["next_config"] = self._compact_cfg(decision.x_next)
        decision.meta["anchor_meta"] = self.anchor_tracker.to_dict()

        pause_before = int(self.post_cold_pause_remaining)
        if pause_before > 0:
            self.post_cold_pause_remaining = max(0, pause_before - 1)
        decision.meta["cold_start_post_pause_remaining_before"] = float(pause_before)
        decision.meta["cold_start_post_pause_remaining_after"] = float(
            max(0, int(self.post_cold_pause_remaining))
        )

        # 9) Update dwell/state and set x_prev.
        self._update_state_after_decision(decision, derived)
        self._append_history(window_metrics)
        self.prev_cold_start_active = bool(cold_start_active)
        self.total_steps += 1
        self.last_decision = decision
        return decision

    @staticmethod
    def _contains_config(candidates: Sequence[ConfigX], target: ConfigX) -> bool:
        key = target.hash_key()
        return any(cfg.hash_key() == key for cfg in candidates)

    def _anchor_protected_pool(
        self,
        *,
        safe_set: Sequence[ConfigX],
        anchor_cfg: Optional[ConfigX],
        radius: int,
    ) -> List[ConfigX]:
        if not safe_set:
            return [self.x_prev]
        if anchor_cfg is None:
            return list(safe_set)

        step = max(1, int(radius))
        anchor_key = anchor_cfg.hash_key()
        out: List[ConfigX] = []
        for cfg in safe_set:
            if self._switch_distance(cfg, anchor_cfg) <= float(step):
                out.append(cfg)

        if not out:
            for cfg in safe_set:
                if cfg.hash_key() == anchor_key:
                    out.append(cfg)
                    break
        if not out and self._contains_config(safe_set, self.x_prev):
            out = [self.x_prev]
        if not out and safe_set:
            out = [safe_set[0]]
        return out

    def _trusted_subset(
        self,
        candidates: Sequence[ConfigX],
    ) -> Tuple[List[ConfigX], Dict[str, Dict[str, float]]]:
        trusted: List[ConfigX] = []
        detail: Dict[str, Dict[str, float]] = {}
        now = int(self.stats_store.current_tick())
        min_count = int(max(0, self.cfg.min_count_for_trust))
        max_stale = None if self.cfg.max_staleness_ticks is None else int(max(0, self.cfg.max_staleness_ticks))

        for cfg in candidates:
            key = cfg.hash_key()
            last_seen, count = self.stats_store.get_last_seen_count(self.regime_id, cfg)
            staleness = None if last_seen is None else max(0, now - int(last_seen))
            trust_by_count = int(count) >= min_count
            trust_by_recency = True
            if max_stale is not None:
                trust_by_recency = staleness is not None and int(staleness) <= max_stale
            is_trusted = bool(trust_by_count and trust_by_recency)
            if is_trusted:
                trusted.append(cfg)
            detail[key] = {
                "count": float(count),
                "last_seen": -1.0 if last_seen is None else float(last_seen),
                "staleness": -1.0 if staleness is None else float(staleness),
                "is_trusted": 1.0 if is_trusted else 0.0,
            }
        return trusted, detail

    def _force_adapt_reasons(
        self,
        derived: DerivedMetrics,
        *,
        sample_count: int = 0,
        warmup_active: bool = False,
    ) -> List[str]:
        if warmup_active:
            return []
        reasons: List[str] = []
        quality_gate = int(sample_count) >= max(
            int(self.cfg.min_count_for_trust),
            int(max(1, int(self.cfg.force_adapt_min_count))),
        )
        if derived.quality_margin is not None and float(derived.quality_margin) <= float(
            self.cfg.force_adapt_quality_margin
        ) and quality_gate:
            reasons.append("quality_margin_low")
        if derived.latency_margin is not None and float(derived.latency_margin) <= float(
            self.cfg.force_adapt_latency_margin
        ):
            reasons.append("latency_margin_low")
        return reasons

    def _adapt_exit_ready(self, derived: DerivedMetrics) -> bool:
        quality_ok = (
            derived.quality_margin is None
            or float(derived.quality_margin) >= float(self.cfg.adapt_exit_quality_margin)
        )
        latency_ok = (
            derived.latency_margin is None
            or float(derived.latency_margin) >= float(self.cfg.adapt_exit_latency_margin)
        )
        mem_ok = derived.mem_margin is None or float(derived.mem_margin) >= 0.0
        train_ok = derived.train_tps_margin is None or float(derived.train_tps_margin) >= 0.0
        return bool(quality_ok and latency_ok and mem_ok and train_ok)

    def _should_force_quality_salvage(
        self,
        derived: DerivedMetrics,
        safe: SafetyResult,
        *,
        sample_count: int = 0,
        warmup_active: bool = False,
    ) -> bool:
        if warmup_active:
            return False
        if int(sample_count) < max(
            int(self.cfg.min_count_for_trust),
            int(max(1, int(self.cfg.force_adapt_min_count))),
        ):
            return False
        streak = int(self.dwell_state.get("infeasible_streak", 0))
        threshold = max(1, int(self.cfg.salvage_infeasible_streak))
        quality_bad = derived.quality_margin is not None and float(derived.quality_margin) < 0.0
        return bool(streak >= threshold and (quality_bad or bool(safe.salvaged)))

    def _cold_start_preferred_axis(self) -> str:
        if not bool(self.cfg.cold_start_axis_rotation):
            return "any"
        i_span = max(1, int(self.cfg.cold_start_i_major_span))
        cycle = i_span + 1
        slot = int(self.cold_probe_count) % cycle
        if slot < i_span:
            return "inf"
        return "ctx"

    @staticmethod
    def _probe_abort_due_to_violation(derived: DerivedMetrics) -> bool:
        for margin in (
            derived.quality_margin,
            derived.latency_margin,
            derived.mem_margin,
            derived.train_tps_margin,
        ):
            if margin is not None and float(margin) < 0.0:
                return True
        return False

    @staticmethod
    def _probe_abort_for_reroute(derived: DerivedMetrics) -> bool:
        # REROUTE may be triggered precisely by latency pressure, so do not
        # block only because latency margin is negative.
        for margin in (
            derived.quality_margin,
            derived.mem_margin,
            derived.train_tps_margin,
        ):
            if margin is not None and float(margin) < 0.0:
                return True
        return False

    @staticmethod
    def _switch_distance(a: ConfigX, b: ConfigX) -> float:
        distance = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
        if a.ib != b.ib:
            distance += 2.0
        if a.tb != b.tb:
            distance += 2.0
        return float(distance)

    @staticmethod
    def _is_probe_reason(reason: str) -> bool:
        r = str(reason).lower()
        return ("probe" in r) or ("seek_anchor" in r) or ("reroute_trial" in r)

    def _is_non_cold_single_step(self, cfg: ConfigX, *, max_fast_step: int = 1) -> bool:
        step = max(1, int(max_fast_step))
        return (
            abs(int(cfg.ctx) - int(self.x_prev.ctx)) <= step
            and abs(int(cfg.inf) - int(self.x_prev.inf)) <= step
        )

    def _compute_cold_start_active(self) -> bool:
        if self.cold_start_terminated:
            return False
        min_windows = int(max(0, self.cfg.cold_start_windows))
        max_windows = int(max(min_windows, self.cfg.cold_start_max_windows))
        if self.total_steps < min_windows:
            return True
        patience = int(max(1, self.cfg.cold_start_patience_directions))
        enough_probe_obs = self.cold_start_probe_observations >= patience
        if enough_probe_obs and self.cold_start_no_improve_streak >= patience:
            self.cold_start_terminated = True
            return False
        if self.total_steps >= max_windows:
            self.cold_start_terminated = True
            return False
        return True

    def _update_local_relax_state(
        self,
        *,
        mode: str,
        warmup_active: bool,
        cold_start_active: bool,
        drift_signal: float,
    ) -> bool:
        if not bool(self.cfg.local_relax_enable):
            self.local_relax_remaining = 0
            return False

        if warmup_active or cold_start_active:
            self.local_relax_remaining = 0
            return False

        if str(mode).upper() != "ADAPT":
            self.local_relax_remaining = 0
            return False

        threshold = float(max(0.0, self.cfg.local_relax_drift_threshold))
        budget = int(max(0, self.cfg.local_relax_windows))
        if budget <= 0:
            self.local_relax_remaining = 0
            return False

        if float(drift_signal) >= threshold:
            self.local_relax_remaining = max(self.local_relax_remaining, budget)
        elif self.local_relax_remaining > 0:
            self.local_relax_remaining -= 1

        return self.local_relax_remaining > 0

    def _augment_safe_set_local_relax(
        self,
        *,
        safe_set: Sequence[ConfigX],
        candidates: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
    ) -> List[ConfigX]:
        if not candidates:
            return list(safe_set)

        radius = max(1, int(self.cfg.local_relax_radius))
        local = self.config_space.neighbors(
            self.x_prev,
            eps_fast=radius,
            eps_slow=0,
        )
        local_keys = {cfg.hash_key() for cfg in local}
        prev_key = self.x_prev.hash_key()

        q_slack = float(max(0.0, self.cfg.local_relax_quality_slack))
        lat_slack = float(max(0.0, self.cfg.local_relax_latency_slack))

        merged = list(safe_set)
        seen = {cfg.hash_key() for cfg in merged}

        for cfg in candidates:
            key = cfg.hash_key()
            if key in seen or key == prev_key or key not in local_keys:
                continue
            row = dict(margin_report.get(key, {}))
            if not row:
                continue

            mem_margin = float(row.get("mem_margin", 0.0))
            train_margin = float(row.get("train_tps_margin", 0.0))
            q_margin = float(row.get("quality_margin", 0.0))
            l_margin = float(row.get("latency_margin", 0.0))

            if mem_margin < 0.0 or train_margin < 0.0:
                continue
            if q_margin < -q_slack:
                continue
            if l_margin < -lat_slack:
                continue

            merged.append(cfg)
            seen.add(key)

        return merged

    def _update_cold_start_progress(self, window: WindowMetrics) -> None:
        # Current window observes the config chosen by the previous decision.
        last = self.last_decision
        if last is None:
            return
        meta = dict(last.meta or {})
        if not bool(meta.get("cold_start_active", False)):
            return
        if not self._is_probe_reason(last.reason):
            return

        latency_ref = float(window.lat_p95) if window.lat_p95 is not None else float(window.lat_mean)
        quality_ref = float(window.acc_mean)
        lat_eps = float(max(0.0, self.cfg.cold_start_improve_latency_eps))
        q_eps = float(max(0.0, self.cfg.cold_start_improve_quality_eps))

        improved_latency = False
        improved_quality = False

        if self.cold_start_best_latency is None:
            improved_latency = True
            self.cold_start_best_latency = latency_ref
        elif latency_ref < float(self.cold_start_best_latency) - lat_eps:
            improved_latency = True
            self.cold_start_best_latency = latency_ref
        else:
            self.cold_start_best_latency = min(float(self.cold_start_best_latency), latency_ref)

        if self.cold_start_best_quality is None:
            improved_quality = True
            self.cold_start_best_quality = quality_ref
        elif quality_ref > float(self.cold_start_best_quality) + q_eps:
            improved_quality = True
            self.cold_start_best_quality = quality_ref
        else:
            self.cold_start_best_quality = max(float(self.cold_start_best_quality), quality_ref)

        self.cold_start_probe_observations += 1
        if improved_latency or improved_quality:
            self.cold_start_no_improve_streak = 0
        else:
            self.cold_start_no_improve_streak += 1

    def _seek_exit_ready(self, derived: DerivedMetrics) -> bool:
        anchor = self.anchor_tracker.state.anchor_cfg
        if anchor is None:
            return False
        stable_windows = int(max(0, self.cfg.seek_exit_stable_windows))
        if stable_windows > 0 and int(self.seek_anchor_stable_streak) >= stable_windows:
            return True
        conf = float(self.anchor_tracker.state.anchor_confidence)
        if conf < float(max(0.0, min(1.0, self.cfg.seek_exit_anchor_confidence))):
            return False
        min_probe_obs = int(max(0, self.cfg.seek_exit_min_probe_observations))
        if int(self.cold_start_probe_observations) < min_probe_obs:
            return False
        return True

    def _window_objective_score(self, window: WindowMetrics) -> float:
        return float(
            self._preference_score(
                self.cfg.preference,
                float(window.lat_mean),
                float(window.acc_mean),
            )
        )

    def _estimated_objective_score(self, cfg: ConfigX, window: WindowMetrics) -> float:
        lat, acc = self._mu_pair(cfg, window)
        return float(self._preference_score(self.cfg.preference, lat, acc))

    def _reset_jsd_probe_state(self, reason: str = "") -> None:
        self.jsd_probe_state = {}
        if reason:
            self.jsd_probe_state["_last_reset_reason"] = reason

    def _jsd_probe_axis_order(self) -> List[str]:
        raw = str(getattr(self.cfg, "jsd_probe_axis_order", "ctx,inf") or "ctx,inf")
        order: List[str] = []
        for token in raw.split(","):
            axis = token.strip().lower()
            if axis not in ("ctx", "inf"):
                continue
            if axis not in order:
                order.append(axis)
        if not order:
            order = ["ctx", "inf"]
        if "ctx" not in order:
            order.append("ctx")
        if "inf" not in order:
            order.append("inf")
        return order

    def _jsd_probe_invalidate_neighborhood(self, base_cfg: ConfigX, state: Dict[str, Any]) -> None:
        if not bool(getattr(self.cfg, "jsd_probe_force_refresh", True)):
            return
        radius = max(1, int(getattr(self.cfg, "jsd_probe_invalidate_radius", 3)))
        hold_ticks = max(1, int(getattr(self.cfg, "jsd_probe_invalidate_hold_ticks", 256)))
        local = self.config_space.neighbors(base_cfg, eps_fast=radius, eps_slow=0)
        base_key = base_cfg.hash_key()
        to_invalidate = [cfg for cfg in local if cfg.hash_key() != base_key]
        if not to_invalidate:
            return
        self.stats_store.invalidate_many(
            self.regime_id,
            to_invalidate,
            hold_ticks=hold_ticks,
            reset_count=True,
            reset_ema=False,
        )
        invalidated = set(state.get("invalidated_keys", []))
        invalidated.update(cfg.hash_key() for cfg in to_invalidate)
        state["invalidated_keys"] = sorted(invalidated)
        state["invalidate_radius"] = float(radius)
        state["invalidate_hold_ticks"] = float(hold_ticks)

    def _jsd_probe_candidate_pool(
        self,
        *,
        base_cfg: ConfigX,
        safe_set: Sequence[ConfigX],
        candidates: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        state: Mapping[str, Any],
    ) -> List[ConfigX]:
        radius = max(1, int(getattr(self.cfg, "jsd_probe_invalidate_radius", 3)))
        local = self.config_space.neighbors(base_cfg, eps_fast=radius, eps_slow=0)
        base_key = base_cfg.hash_key()
        local_map: Dict[str, ConfigX] = {base_key: base_cfg}
        for cfg in local:
            if cfg.ib == base_cfg.ib and cfg.tb == base_cfg.tb:
                local_map[cfg.hash_key()] = cfg

        safe_keys = {cfg.hash_key() for cfg in safe_set}
        invalidated_keys = set(state.get("invalidated_keys", []))
        pool: List[ConfigX] = [base_cfg]
        seen = {base_key}

        def _allow_by_margin(cfg_key: str) -> bool:
            row = dict(margin_report.get(cfg_key, {}))
            if not row:
                return True
            mem_margin = float(row.get("mem_margin", 0.0))
            train_margin = float(row.get("train_tps_margin", 0.0))
            return mem_margin >= -0.15 and train_margin >= -0.15

        for cfg in candidates:
            key = cfg.hash_key()
            if key in seen:
                continue
            local_cfg = local_map.get(key)
            if local_cfg is None:
                continue
            if not _allow_by_margin(key):
                continue
            if (
                key in safe_keys
                or key in invalidated_keys
                or self.stats_store.is_invalid(self.regime_id, local_cfg)
            ):
                pool.append(local_cfg)
                seen.add(key)

        for cfg in safe_set:
            key = cfg.hash_key()
            if key in seen:
                continue
            local_cfg = local_map.get(key)
            if local_cfg is None:
                continue
            if not _allow_by_margin(key):
                continue
            pool.append(local_cfg)
            seen.add(key)

        return pool

    def _jsd_probe_pick_axis_candidate(
        self,
        *,
        base_cfg: ConfigX,
        pool: Sequence[ConfigX],
        axis: str,
        direction: int,
        max_step: int = 1,
    ) -> Optional[ConfigX]:
        if not pool or int(direction) == 0:
            return None
        axis_name = str(axis).lower()
        if axis_name not in ("ctx", "inf"):
            return None
        step_cap = max(1, int(max_step))
        base_ctx = int(base_cfg.ctx)
        base_inf = int(base_cfg.inf)
        base_key = base_cfg.hash_key()

        def _delta(cfg: ConfigX) -> int:
            return int(cfg.ctx) - base_ctx if axis_name == "ctx" else int(cfg.inf) - base_inf

        filtered: List[ConfigX] = []
        for cfg in pool:
            if cfg.hash_key() == base_key:
                continue
            if axis_name == "ctx":
                if int(cfg.inf) != base_inf:
                    continue
            else:
                if int(cfg.ctx) != base_ctx:
                    continue
            delta = _delta(cfg)
            if int(direction) > 0 and delta <= 0:
                continue
            if int(direction) < 0 and delta >= 0:
                continue
            if abs(int(delta)) > step_cap:
                continue
            filtered.append(cfg)

        if not filtered:
            return None

        return min(
            filtered,
            key=lambda cfg: (
                abs(abs(int(_delta(cfg))) - 1),
                abs(int(_delta(cfg))),
                self._switch_distance(cfg, base_cfg),
                int(cfg.ctx),
                int(cfg.inf),
            ),
        )

    def _jsd_probe_advance_axis_or_finish(
        self,
        *,
        state: Dict[str, Any],
        next_base_cfg: ConfigX,
        next_base_score: float,
    ) -> None:
        axis_order = list(state.get("axis_order", self._jsd_probe_axis_order()))
        axis_idx = int(state.get("axis_idx", 0)) + 1
        if axis_idx >= len(axis_order):
            self._reset_jsd_probe_state("committed")
            return
        state["axis_idx"] = int(axis_idx)
        state["axis"] = str(axis_order[axis_idx])
        state["stage"] = "probe_up"
        state["base_cfg"] = next_base_cfg
        state["base_score"] = float(next_base_score)
        state["down_cfg"] = None
        state["down_score"] = None
        state["up_cfg"] = None
        state["up_score"] = None
        state["continue_dir"] = 0
        state["continue_remaining"] = 0
        state["continue_base_cfg"] = None
        state["continue_base_score"] = None
        state["continue_cfg"] = None
        state["continue_score"] = None
        self._jsd_probe_invalidate_neighborhood(next_base_cfg, state)
        self.jsd_probe_state = state

    def _update_jsd_probe_observation(self, window: WindowMetrics) -> None:
        state = self.jsd_probe_state or {}
        if not state or not bool(state.get("active", False)):
            return
        last = self.last_decision
        if last is None:
            return
        reason = str(last.reason)
        observed = self._window_objective_score(window)
        if reason == "jsd_probe_up":
            up_cfg = state.get("up_cfg")
            if isinstance(up_cfg, ConfigX) and up_cfg.hash_key() == last.x_next.hash_key():
                state["up_score"] = float(observed)
                state["stage"] = "probe_down"
        elif reason == "jsd_probe_down":
            down_cfg = state.get("down_cfg")
            if isinstance(down_cfg, ConfigX) and down_cfg.hash_key() == last.x_next.hash_key():
                state["down_score"] = float(observed)
                state["stage"] = "commit"
        elif reason == "jsd_probe_continue":
            continue_cfg = state.get("continue_cfg")
            if isinstance(continue_cfg, ConfigX) and continue_cfg.hash_key() == last.x_next.hash_key():
                state["continue_score"] = float(observed)
                state["stage"] = "continue_commit"
        self.jsd_probe_state = state

    def _maybe_jsd_probe_decision(
        self,
        *,
        safe_set: Sequence[ConfigX],
        candidates: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        window: WindowMetrics,
        derived: DerivedMetrics,
        local_relax_active: bool,
    ) -> Optional[Decision]:
        if not bool(self.cfg.jsd_local_probe_enable):
            self._reset_jsd_probe_state("disabled")
            return None
        if self.phase_machine.state.phase != Phase.SHOCK_LOCAL_REROUTE:
            self._reset_jsd_probe_state("phase_not_shock")
            return None

        state = self.jsd_probe_state or {}
        hold_windows = max(0, int(getattr(self.cfg, "jsd_probe_state_hold_windows", 3)))
        drift_threshold = float(max(0.0, self.cfg.local_relax_drift_threshold))
        jsd_trigger_now = float(derived.shock_drift) >= drift_threshold

        if not bool(local_relax_active):
            if state and bool(state.get("active", False)) and hold_windows > 0:
                remaining = int(state.get("idle_keep_remaining", hold_windows))
                if remaining > 0:
                    state["idle_keep_remaining"] = int(remaining - 1)
                    self.jsd_probe_state = state
                    return None
            self._reset_jsd_probe_state("local_relax_inactive")
            return None

        # Keep an active probe session for a short grace period even when
        # the current window does not cross the JSD trigger threshold.
        if not jsd_trigger_now:
            if state and bool(state.get("active", False)) and hold_windows > 0:
                remaining = int(state.get("idle_keep_remaining", hold_windows))
                if remaining > 0:
                    state["idle_keep_remaining"] = int(remaining - 1)
                    self.jsd_probe_state = state
                    return None
                self._reset_jsd_probe_state("no_new_jsd_timeout")
                return None
            self._reset_jsd_probe_state("no_new_jsd")
            return None

        if not state or not bool(state.get("active", False)):
            axis_order = self._jsd_probe_axis_order()
            state = {
                "active": True,
                "stage": "probe_up",
                "start_step": int(self.total_steps),
                "axis_order": axis_order,
                "axis_idx": 0,
                "axis": str(axis_order[0]),
                "base_cfg": self.x_prev,
                "base_score": float(self._window_objective_score(window)),
                "down_cfg": None,
                "down_score": None,
                "up_cfg": None,
                "up_score": None,
                "continue_dir": 0,
                "continue_remaining": 0,
                "continue_base_cfg": None,
                "continue_base_score": None,
                "continue_cfg": None,
                "continue_score": None,
                "invalidated_keys": [],
                "idle_keep_remaining": int(hold_windows),
            }
            self._jsd_probe_invalidate_neighborhood(self.x_prev, state)
            self.jsd_probe_state = state
        else:
            state["idle_keep_remaining"] = int(hold_windows)
            self.jsd_probe_state = state

        for _ in range(8):
            stage = str(state.get("stage", "probe_up"))
            base_cfg = state.get("base_cfg")
            if not isinstance(base_cfg, ConfigX):
                self._reset_jsd_probe_state("invalid_base_cfg")
                return None
            axis = str(state.get("axis", "ctx")).lower()
            probe_pool = self._jsd_probe_candidate_pool(
                base_cfg=base_cfg,
                safe_set=safe_set,
                candidates=candidates,
                margin_report=margin_report,
                state=state,
            )

            if stage == "probe_down":
                down_cfg = self._jsd_probe_pick_axis_candidate(
                    base_cfg=base_cfg,
                    pool=probe_pool,
                    axis=axis,
                    direction=-1,
                    max_step=1,
                )
                if down_cfg is None:
                    state["stage"] = "commit"
                    self.jsd_probe_state = state
                    continue
                state["down_cfg"] = down_cfg
                self.jsd_probe_state = state
                return Decision(
                    x_next=down_cfg,
                    reason="jsd_probe_down",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={
                        "jsd_probe_stage": "probe_down",
                        "jsd_probe_axis": axis,
                        "jsd_probe_base_cfg": self._compact_cfg(base_cfg),
                    },
                )

            if stage == "probe_up":
                up_cfg = self._jsd_probe_pick_axis_candidate(
                    base_cfg=base_cfg,
                    pool=probe_pool,
                    axis=axis,
                    direction=1,
                    max_step=1,
                )
                if up_cfg is None:
                    state["stage"] = "probe_down"
                    self.jsd_probe_state = state
                    continue
                state["up_cfg"] = up_cfg
                self.jsd_probe_state = state
                return Decision(
                    x_next=up_cfg,
                    reason="jsd_probe_up",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={
                        "jsd_probe_stage": "probe_up",
                        "jsd_probe_axis": axis,
                        "jsd_probe_base_cfg": self._compact_cfg(base_cfg),
                    },
                )

            if stage == "continue_probe":
                remaining = int(state.get("continue_remaining", 0))
                if remaining <= 0:
                    state["stage"] = "continue_commit"
                    self.jsd_probe_state = state
                    continue
                continue_base = state.get("continue_base_cfg")
                if not isinstance(continue_base, ConfigX):
                    state["stage"] = "continue_commit"
                    self.jsd_probe_state = state
                    continue
                continue_pool = self._jsd_probe_candidate_pool(
                    base_cfg=continue_base,
                    safe_set=safe_set,
                    candidates=candidates,
                    margin_report=margin_report,
                    state=state,
                )
                dir_sign = int(state.get("continue_dir", 0))
                continue_cfg = self._jsd_probe_pick_axis_candidate(
                    base_cfg=continue_base,
                    pool=continue_pool,
                    axis=axis,
                    direction=dir_sign,
                    max_step=1,
                )
                if continue_cfg is None:
                    state["stage"] = "continue_commit"
                    self.jsd_probe_state = state
                    continue
                state["continue_cfg"] = continue_cfg
                self.jsd_probe_state = state
                return Decision(
                    x_next=continue_cfg,
                    reason="jsd_probe_continue",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={
                        "jsd_probe_stage": "continue_probe",
                        "jsd_probe_axis": axis,
                        "jsd_probe_base_cfg": self._compact_cfg(continue_base),
                        "jsd_probe_continue_dir": float(dir_sign),
                        "jsd_probe_continue_remaining": float(remaining),
                    },
                )

            if stage == "continue_commit":
                continue_base = state.get("continue_base_cfg")
                continue_base_score = state.get("continue_base_score")
                continue_cfg = state.get("continue_cfg")
                continue_score = state.get("continue_score")
                options: List[Tuple[ConfigX, float, str]] = []
                if isinstance(continue_base, ConfigX) and isinstance(continue_base_score, (int, float)):
                    options.append((continue_base, float(continue_base_score), "base"))
                if isinstance(continue_cfg, ConfigX):
                    score = (
                        float(continue_score)
                        if isinstance(continue_score, (int, float))
                        else float(self._estimated_objective_score(continue_cfg, window))
                    )
                    options.append((continue_cfg, score, "cont"))
                if not options:
                    state["continue_remaining"] = 0
                    self._jsd_probe_advance_axis_or_finish(
                        state=state,
                        next_base_cfg=base_cfg,
                        next_base_score=float(state.get("base_score", self._window_objective_score(window))),
                    )
                    return Decision(
                        x_next=base_cfg,
                        reason="jsd_probe_commit",
                        mode=self.mode,
                        regime_id=self.regime_id,
                        meta={
                            "jsd_probe_stage": "continue_commit",
                            "jsd_probe_axis": axis,
                            "jsd_probe_pick": "base",
                        },
                    )
                best_cfg, best_score, best_tag = min(options, key=lambda item: float(item[1]))
                remaining = max(0, int(state.get("continue_remaining", 0)) - 1)
                if best_tag == "cont" and remaining > 0:
                    state["stage"] = "continue_probe"
                    state["continue_remaining"] = int(remaining)
                    state["continue_base_cfg"] = best_cfg
                    state["continue_base_score"] = float(best_score)
                    state["continue_cfg"] = None
                    state["continue_score"] = None
                    self.jsd_probe_state = state
                else:
                    state["continue_remaining"] = 0
                    self._jsd_probe_advance_axis_or_finish(
                        state=state,
                        next_base_cfg=best_cfg,
                        next_base_score=float(best_score),
                    )
                return Decision(
                    x_next=best_cfg,
                    reason="jsd_probe_commit",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={
                        "jsd_probe_stage": "continue_commit",
                        "jsd_probe_axis": axis,
                        "jsd_probe_pick": str(best_tag),
                        "jsd_probe_best_score": float(best_score),
                    },
                )

            if stage != "commit":
                self._reset_jsd_probe_state("invalid_stage")
                return None

            options: List[Tuple[ConfigX, float, str]] = []
            base_score = state.get("base_score")
            if isinstance(base_score, (int, float)):
                options.append((base_cfg, float(base_score), "base"))

            down_cfg = state.get("down_cfg")
            down_score = state.get("down_score")
            if isinstance(down_cfg, ConfigX):
                score = (
                    float(down_score)
                    if isinstance(down_score, (int, float))
                    else float(self._estimated_objective_score(down_cfg, window))
                )
                options.append((down_cfg, score, "down"))

            up_cfg = state.get("up_cfg")
            up_score = state.get("up_score")
            if isinstance(up_cfg, ConfigX):
                score = (
                    float(up_score)
                    if isinstance(up_score, (int, float))
                    else float(self._estimated_objective_score(up_cfg, window))
                )
                options.append((up_cfg, score, "up"))

            if not options:
                self._reset_jsd_probe_state("no_options")
                return None

            best_cfg, best_score, best_tag = min(options, key=lambda item: float(item[1]))
            best_dir = -1 if best_tag == "down" else (1 if best_tag == "up" else 0)
            continue_steps = max(0, int(getattr(self.cfg, "jsd_probe_continue_steps", 1)))
            if best_dir != 0 and continue_steps > 0:
                state["stage"] = "continue_probe"
                state["continue_dir"] = int(best_dir)
                state["continue_remaining"] = int(continue_steps)
                state["continue_base_cfg"] = best_cfg
                state["continue_base_score"] = float(best_score)
                state["continue_cfg"] = None
                state["continue_score"] = None
                self.jsd_probe_state = state
            else:
                self._jsd_probe_advance_axis_or_finish(
                    state=state,
                    next_base_cfg=best_cfg,
                    next_base_score=float(best_score),
                )

            return Decision(
                x_next=best_cfg,
                reason="jsd_probe_commit",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={
                    "jsd_probe_stage": "commit",
                    "jsd_probe_axis": axis,
                    "jsd_probe_pick": str(best_tag),
                    "jsd_probe_best_score": float(best_score),
                    "jsd_probe_continue_dir": float(best_dir),
                    "jsd_probe_candidates": [
                        {
                            "tag": tag,
                            "score": float(score),
                            "cfg": self._compact_cfg(cfg),
                        }
                        for cfg, score, tag in options
                    ],
                    "jsd_probe_invalidated_count": float(len(state.get("invalidated_keys", []))),
                },
            )

        self._reset_jsd_probe_state("loop_guard")
        return None

    def _reroute_guard_hold_this_step(self) -> bool:
        cooldown = int(max(0.0, float(self.reroute_guard_state.get("cooldown", 0.0))))
        if cooldown <= 0:
            return False
        self.reroute_guard_state["cooldown"] = float(cooldown - 1)
        return True

    def _update_reroute_guard(
        self,
        *,
        phase: Phase,
        decision: Decision,
        attempted: bool,
    ) -> None:
        if phase != Phase.SHOCK_LOCAL_REROUTE:
            self.reroute_guard_state["fail_streak"] = 0.0
            return
        if not attempted:
            return

        attempts = float(self.reroute_guard_state.get("attempts", 0.0)) + 1.0
        success = float(self.reroute_guard_state.get("success", 0.0))
        fail_streak = float(self.reroute_guard_state.get("fail_streak", 0.0))

        if str(decision.reason) == "local_reroute_trial":
            success += 1.0
            fail_streak = 0.0
        else:
            fail_streak += 1.0

        self.reroute_guard_state["attempts"] = attempts
        self.reroute_guard_state["success"] = success
        self.reroute_guard_state["fail_streak"] = fail_streak

        min_attempts = max(1, int(self.cfg.reroute_guard_min_attempts))
        if attempts < float(min_attempts):
            return
        success_rate = success / max(1.0, attempts)
        min_success = float(max(0.0, min(1.0, self.cfg.reroute_guard_min_success_rate)))
        fail_threshold = float(max(1, int(self.cfg.reroute_guard_fail_streak)))

        if success_rate >= min_success and fail_streak < fail_threshold:
            # Keep a rolling horizon to avoid unbounded counters.
            if attempts >= 64.0:
                self.reroute_guard_state["attempts"] = attempts * 0.5
                self.reroute_guard_state["success"] = success * 0.5
            return

        self.reroute_guard_state["cooldown"] = float(
            max(
                float(self.reroute_guard_state.get("cooldown", 0.0)),
                max(1, int(self.cfg.reroute_guard_cooldown_windows)),
            )
        )
        self.reroute_guard_state["attempts"] = 0.0
        self.reroute_guard_state["success"] = 0.0
        self.reroute_guard_state["fail_streak"] = 0.0

    def _reroute_guard_meta(self) -> Dict[str, float]:
        attempts = float(self.reroute_guard_state.get("attempts", 0.0))
        success = float(self.reroute_guard_state.get("success", 0.0))
        return {
            "attempts": attempts,
            "success": success,
            "success_rate": 0.0 if attempts <= 0.0 else success / attempts,
            "fail_streak": float(self.reroute_guard_state.get("fail_streak", 0.0)),
            "cooldown_remaining": float(self.reroute_guard_state.get("cooldown", 0.0)),
            "min_attempts": float(max(1, int(self.cfg.reroute_guard_min_attempts))),
            "min_success_rate": float(
                max(0.0, min(1.0, float(self.cfg.reroute_guard_min_success_rate)))
            ),
            "fail_streak_threshold": float(max(1, int(self.cfg.reroute_guard_fail_streak))),
            "cooldown_windows": float(max(1, int(self.cfg.reroute_guard_cooldown_windows))),
        }

    def _apply_non_cold_step_cap(
        self,
        *,
        decision: Decision,
        safe_set: Sequence[ConfigX],
        window: WindowMetrics,
    ) -> Decision:
        # Hard guard: phase-aware fast-knob step cap.
        phase = self.phase_machine.state.phase
        max_non_cold_step = max_fast_step_for_phase(
            phase,
            hold_radius=max(1, int(self.cfg.hold_anchor_radius)),
            reroute_radius=max(1, int(self.cfg.reroute_radius)),
            seek_radius=max(1, int(self.cfg.cold_start_probe_eps_fast)),
        )
        if self._is_non_cold_single_step(decision.x_next, max_fast_step=max_non_cold_step):
            return decision

        target = decision.x_next
        dctx_target = int(target.ctx) - int(self.x_prev.ctx)
        dinf_target = int(target.inf) - int(self.x_prev.inf)

        def _sgn(v: int) -> int:
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0

        s_ctx = _sgn(dctx_target)
        s_inf = _sgn(dinf_target)

        pool = [
            cfg
            for cfg in safe_set
            if cfg.hash_key() != self.x_prev.hash_key()
            and self._is_non_cold_single_step(cfg, max_fast_step=max_non_cold_step)
            and cfg.ib == self.x_prev.ib
            and cfg.tb == self.x_prev.tb
        ]
        target_is_non_ascending = (
            int(target.ctx) <= int(self.x_prev.ctx)
            and int(target.inf) <= int(self.x_prev.inf)
        )
        if target_is_non_ascending:
            directed_pool = [
                cfg
                for cfg in pool
                if int(cfg.ctx) <= int(self.x_prev.ctx) and int(cfg.inf) <= int(self.x_prev.inf)
            ]
            if directed_pool:
                pool = directed_pool
        if not pool:
            return Decision(
                x_next=self.x_prev,
                reason="non_cold_step_cap_hold",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={
                    "blocked_reason": decision.reason,
                    "max_non_cold_step": float(max_non_cold_step),
                    "requested_config": self._compact_cfg(decision.x_next),
                },
            )

        def _axis_dir_score(delta: int, target_sign: int) -> int:
            if target_sign == 0:
                return 0
            if delta == 0:
                return 0
            if _sgn(delta) == target_sign:
                return 1
            return -1

        def _score(cfg: ConfigX) -> Tuple[float, ...]:
            dctx = int(cfg.ctx) - int(self.x_prev.ctx)
            dinf = int(cfg.inf) - int(self.x_prev.inf)
            dir_score = _axis_dir_score(dctx, s_ctx) + _axis_dir_score(dinf, s_inf)
            target_dist = self._switch_distance(cfg, target)
            gain = self._predicted_switch_gain(cfg, window)
            return (
                float(dir_score),
                -float(target_dist),
                float(gain),
                -self._switch_distance(cfg, self.x_prev),
            )

        capped = max(pool, key=_score)
        meta = dict(decision.meta or {})
        meta["non_cold_step_cap_applied"] = 1.0
        meta["non_cold_step_cap_from"] = self._compact_cfg(decision.x_next)
        meta["non_cold_step_cap_to"] = self._compact_cfg(capped)
        meta["max_non_cold_step"] = float(max_non_cold_step)
        return Decision(
            x_next=capped,
            reason=decision.reason,
            predicted_gains=dict(decision.predicted_gains or {}),
            safety_margins=dict(decision.safety_margins or {}),
            regime_id=decision.regime_id,
            mode=decision.mode,
            meta=meta,
        )

    def _phase_fallback_decision(
        self,
        *,
        phase: Phase,
        safe_set: Sequence[ConfigX],
        anchor_cfg: Optional[ConfigX],
        allow_local_uphill: bool = False,
    ) -> Decision:
        if phase == Phase.SHOCK_LOCAL_REROUTE:
            descending = [
                cfg
                for cfg in safe_set
                if cfg.hash_key() != self.x_prev.hash_key()
                and int(cfg.ctx) <= int(self.x_prev.ctx)
                and int(cfg.inf) <= int(self.x_prev.inf)
                and (int(cfg.ctx) < int(self.x_prev.ctx) or int(cfg.inf) < int(self.x_prev.inf))
            ]
            if descending:
                target = min(
                    descending,
                    key=lambda cfg: (
                        self._switch_distance(cfg, self.x_prev),
                        int(cfg.inf),
                        int(cfg.ctx),
                    ),
                )
                return Decision(
                    x_next=target,
                    reason="reroute_fallback_safe_pick",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={"phase_fallback": phase.value},
                )
            if allow_local_uphill:
                radius = max(1, int(self.cfg.local_relax_radius))
                local_pool = [
                    cfg
                    for cfg in safe_set
                    if cfg.hash_key() != self.x_prev.hash_key()
                    and self._switch_distance(cfg, self.x_prev) <= float(radius)
                ]
                if local_pool:
                    target = min(
                        local_pool,
                        key=lambda cfg: (
                            self._switch_distance(cfg, self.x_prev),
                            int(cfg.inf),
                            int(cfg.ctx),
                        ),
                    )
                    return Decision(
                        x_next=target,
                        reason="reroute_fallback_local_relax_pick",
                        mode=self.mode,
                        regime_id=self.regime_id,
                        meta={"phase_fallback": phase.value},
                    )
            if anchor_cfg is not None and self._contains_config(safe_set, anchor_cfg):
                return Decision(
                    x_next=anchor_cfg,
                    reason="reroute_fallback_anchor_hold",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={"phase_fallback": phase.value},
                )
            if self._contains_config(safe_set, self.x_prev):
                return Decision(
                    x_next=self.x_prev,
                    reason="reroute_fallback_prev_hold",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={"phase_fallback": phase.value},
                )
            if safe_set:
                return Decision(
                    x_next=safe_set[0],
                    reason="reroute_fallback_safe_pick",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={"phase_fallback": phase.value},
                )
            return Decision(
                x_next=self.x_safe_default,
                reason="reroute_fallback_safe_default",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={"phase_fallback": phase.value},
            )

        if phase == Phase.SEEK_ANCHOR:
            target = self.x_prev
            if not self._contains_config(safe_set, target):
                if safe_set:
                    target = safe_set[0]
                else:
                    target = self.x_safe_default
            return Decision(
                x_next=target,
                reason="seek_anchor_hold",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={"phase_fallback": phase.value},
            )

        target = self.x_prev
        if not self._contains_config(safe_set, target):
            if safe_set:
                target = safe_set[0]
            else:
                target = self.x_safe_default
        return Decision(
            x_next=target,
            reason="phase_hold",
            mode=self.mode,
            regime_id=self.regime_id,
            meta={"phase_fallback": phase.value},
        )

    def _apply_switch_guards(
        self,
        *,
        decision: Decision,
        safe_set: Sequence[ConfigX],
        window: WindowMetrics,
        derived: DerivedMetrics,
        phase: Phase,
        shock_score: float,
        force_adapt_reasons: Sequence[str],
    ) -> Decision:
        changed = decision.x_next.hash_key() != self.x_prev.hash_key()
        if not changed:
            return decision

        decision = self._apply_non_cold_step_cap(
            decision=decision,
            safe_set=safe_set,
            window=window,
        )
        changed = decision.x_next.hash_key() != self.x_prev.hash_key()
        if not changed:
            return decision

        x_prev_is_safe = self._contains_config(safe_set, self.x_prev)
        hard_violation = self._probe_abort_due_to_violation(derived)
        dwell_remaining = int(self.dwell_state.get("remaining", 0))
        if (
            dwell_remaining > 0
            and x_prev_is_safe
            and not hard_violation
            and not self._is_probe_reason(decision.reason)
        ):
            return Decision(
                x_next=self.x_prev,
                reason="global_dwell_hold",
                mode=self.mode,
                regime_id=self.regime_id,
                meta={
                    "blocked_reason": decision.reason,
                    "dwell_remaining": float(dwell_remaining),
                },
            )

        decision = self._apply_decision_stabilizer(
            decision=decision,
            phase=phase,
            safe_set=safe_set,
            window=window,
            hard_violation=hard_violation,
            x_prev_is_safe=x_prev_is_safe,
            shock_score=shock_score,
            force_adapt_reasons=force_adapt_reasons,
        )
        changed = decision.x_next.hash_key() != self.x_prev.hash_key()
        if not changed:
            return decision

        if (
            not self._is_probe_reason(decision.reason)
            and "salvage" not in decision.reason
            and x_prev_is_safe
            and not hard_violation
            and float(self.cfg.min_switch_gain_eps) > 0.0
        ):
            gain = self._predicted_switch_gain(decision.x_next, window)
            if gain < float(self.cfg.min_switch_gain_eps):
                return Decision(
                    x_next=self.x_prev,
                    reason="min_gain_hold",
                    mode=self.mode,
                    regime_id=self.regime_id,
                    meta={
                        "blocked_reason": decision.reason,
                        "predicted_switch_gain": float(gain),
                        "min_switch_gain_eps": float(self.cfg.min_switch_gain_eps),
                    },
                )
            decision.meta = dict(decision.meta or {})
            decision.meta["predicted_switch_gain"] = float(gain)
            decision.meta["min_switch_gain_eps"] = float(self.cfg.min_switch_gain_eps)
        return decision

    def _apply_decision_stabilizer(
        self,
        *,
        decision: Decision,
        phase: Phase,
        safe_set: Sequence[ConfigX],
        window: WindowMetrics,
        hard_violation: bool,
        x_prev_is_safe: bool,
        shock_score: float,
        force_adapt_reasons: Sequence[str],
    ) -> Decision:
        if not bool(self.cfg.stabilizer_enable):
            return decision
        if hard_violation or not x_prev_is_safe:
            return decision
        if phase in (Phase.WARMUP, Phase.SALVAGE, Phase.SHOCK_LOCAL_REROUTE):
            return decision
        if self.mode == "ADAPT" and not bool(self.cfg.stabilizer_allow_in_adapt):
            return decision
        if float(shock_score) > float(max(0.0, self.cfg.stabilizer_low_shock_max)):
            return decision
        if force_adapt_reasons:
            return decision

        trigger = None
        trigger_meta: Dict[str, Any] = {}
        reason_l = str(decision.reason).lower()

        if bool(self.cfg.stabilizer_two_cycle) and self.x_prev_prev is not None:
            if decision.x_next.hash_key() == self.x_prev_prev.hash_key():
                trigger = "two_cycle"
                trigger_meta["x_prev_prev"] = self._compact_cfg(self.x_prev_prev)

        if (
            trigger is None
            and bool(self.cfg.stabilizer_seek_monotone)
            and phase == Phase.SEEK_ANCHOR
            and "seek_anchor" in reason_l
        ):
            max_rebound = max(0, int(self.cfg.stabilizer_seek_max_inf_rebound))
            rebound = int(decision.x_next.inf) - int(self.x_prev.inf)
            if rebound > max_rebound:
                trigger = "seek_inf_rebound"
                trigger_meta["seek_inf_rebound"] = float(rebound)
                trigger_meta["seek_inf_rebound_cap"] = float(max_rebound)

        low_gain_eps = float(max(0.0, self.cfg.stabilizer_low_shock_gain_eps))
        if trigger is None and low_gain_eps > 0.0:
            gain = float(self._predicted_switch_gain(decision.x_next, window))
            if gain < low_gain_eps:
                trigger = "low_shock_low_gain"
                trigger_meta["predicted_switch_gain"] = gain
                trigger_meta["stabilizer_low_shock_gain_eps"] = low_gain_eps

        if trigger is None:
            return decision

        meta = dict(decision.meta or {})
        meta["stabilizer_trigger"] = trigger
        meta["stabilizer_shadow"] = 1.0 if bool(self.cfg.stabilizer_shadow) else 0.0
        meta["stabilizer_phase"] = phase.value
        meta["stabilizer_shock_score"] = float(shock_score)
        meta.update(trigger_meta)

        if bool(self.cfg.stabilizer_shadow):
            meta["stabilizer_would_hold"] = 1.0
            decision.meta = meta
            return decision

        return Decision(
            x_next=self.x_prev if self._contains_config(safe_set, self.x_prev) else self.x_safe_default,
            reason="stabilizer_hold",
            mode=self.mode,
            regime_id=self.regime_id,
            meta={
                **meta,
                "blocked_reason": decision.reason,
            },
        )

    def _predicted_switch_gain(self, candidate: ConfigX, window: WindowMetrics) -> float:
        prev_lat, prev_acc = self._mu_pair(self.x_prev, window)
        next_lat, next_acc = self._mu_pair(candidate, window)
        prev_score = self._preference_score(self.cfg.preference, prev_lat, prev_acc)
        next_score = self._preference_score(self.cfg.preference, next_lat, next_acc)
        return float(prev_score - next_score)

    def _mu_pair(self, cfg: ConfigX, window: WindowMetrics) -> Tuple[float, float]:
        lat_mu, _ = self.stats_store.get_mu_sigma(self.regime_id, cfg, "lat_mean")
        acc_mu, _ = self.stats_store.get_mu_sigma(self.regime_id, cfg, "acc_mean")
        lat = float(window.lat_mean) if lat_mu is None else float(lat_mu)
        acc = float(window.acc_mean) if acc_mu is None else float(acc_mu)
        return lat, acc

    @staticmethod
    def _preference_score(preference: str, lat: float, quality: float) -> float:
        pref = str(preference).lower()
        if pref in ("latency", "speed", "throughput"):
            return float(lat) - 0.05 * float(quality)
        if pref in ("knee",):
            return float(lat) - float(quality)
        return -float(quality) + 0.05 * float(lat)

    @staticmethod
    def _compact_cfg(cfg: ConfigX) -> Dict[str, Any]:
        return {
            "ctx": int(cfg.ctx),
            "inf": int(cfg.inf),
            "ib": None if cfg.ib is None else int(cfg.ib),
            "tb": None if cfg.tb is None else int(cfg.tb),
            "key": cfg.hash_key(),
        }

    def _derive_metrics(self, window: WindowMetrics) -> DerivedMetrics:
        dm = DerivedMetrics.from_window(
            window,
            tau=self.cfg.tau,
            sla=self.cfg.sla,
            mem_limit=self.cfg.mem_limit,
            train_min=self.cfg.train_min,
        )
        drift_signal = (
            window.jsd_mean
            if window.jsd_mean is not None
            else window.token_drift_mean
        )
        dm.shock_drift = float(drift_signal or 0.0)
        dm.shock_accept = float(window.spec_accept_mean or 0.0)
        dm.shock_reverify = float(window.spec_reverify_per_step or 0.0)
        dm.shock_verify_ratio = float(dm.verify_ratio or 0.0)
        dm.shock_waste_rate = float(dm.waste_rate or 0.0)
        return dm

    @staticmethod
    def _stats_payload(window: WindowMetrics, derived: DerivedMetrics) -> Dict[str, Optional[float]]:
        return {
            "lat_mean": window.lat_mean,
            "lat_p95": window.lat_p95,
            "acc_mean": window.acc_mean,
            "train_tps": window.train_tps,
            "mem_peak": window.mem_peak,
            "verify_ratio": derived.verify_ratio,
            "waste_rate": derived.waste_rate,
        }

    def _mu_vectors(
        self,
        candidates: Sequence[ConfigX],
        window: WindowMetrics,
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for cfg in candidates:
            lat_mu, _ = self.stats_store.get_mu_sigma(self.regime_id, cfg, "lat_mean")
            acc_mu, _ = self.stats_store.get_mu_sigma(self.regime_id, cfg, "acc_mean")
            tps_mu, _ = self.stats_store.get_mu_sigma(self.regime_id, cfg, "train_tps")
            if lat_mu is None:
                lat_mu = window.lat_mean
            if acc_mu is None:
                acc_mu = window.acc_mean
            if tps_mu is None:
                tps_mu = window.train_tps if window.train_tps is not None else 0.0
            out[cfg.hash_key()] = {
                "lat_mean": float(lat_mu),
                "acc_mean": float(acc_mu),
                "train_tps": float(tps_mu),
            }
        return out

    @staticmethod
    def _pareto_configs(
        candidates: Sequence[ConfigX],
        mu_vectors: Mapping[str, Mapping[str, float]],
    ) -> List[ConfigX]:
        if not candidates:
            return []
        vectors = []
        for cfg in candidates:
            mu = mu_vectors.get(cfg.hash_key())
            if not mu:
                continue
            vectors.append((mu["lat_mean"], mu["acc_mean"], mu["train_tps"]))
        keep = set(pareto_filter(vectors, minimize_mask=(True, False, False)))
        out: List[ConfigX] = []
        for cfg in candidates:
            mu = mu_vectors.get(cfg.hash_key())
            if not mu:
                continue
            vec = (mu["lat_mean"], mu["acc_mean"], mu["train_tps"])
            if vec in keep:
                out.append(cfg)
        return out

    @staticmethod
    def _alarm_severity(derived: DerivedMetrics, shock_score: float) -> float:
        sev = 0.0
        if derived.quality_margin is not None and derived.quality_margin < 0:
            sev += min(1.0, abs(float(derived.quality_margin)))
        if derived.latency_margin is not None and derived.latency_margin < 0:
            sev += min(1.0, abs(float(derived.latency_margin)))
        sev += min(1.0, max(0.0, float(shock_score)))
        return min(1.0, sev)

    def _update_state_after_decision(self, decision: Decision, derived: DerivedMetrics) -> None:
        changed = decision.x_next.hash_key() != self.x_prev.hash_key()
        force_dwell_steps = 0
        if decision.meta and "force_dwell_steps" in decision.meta:
            try:
                force_dwell_steps = max(0, int(float(decision.meta["force_dwell_steps"])))
            except Exception:
                force_dwell_steps = 0
        if changed:
            self.dwell_state["remaining"] = max(0, int(self.cfg.dwell_steps), force_dwell_steps)
        elif self.dwell_state.get("remaining", 0) > 0:
            self.dwell_state["remaining"] -= 1
        if force_dwell_steps > int(self.dwell_state.get("remaining", 0)):
            self.dwell_state["remaining"] = force_dwell_steps

        infeasible = False
        if derived.quality_margin is not None and derived.quality_margin < 0:
            infeasible = True
        if derived.latency_margin is not None and derived.latency_margin < 0:
            infeasible = True
        if infeasible:
            self.dwell_state["infeasible_streak"] = int(self.dwell_state.get("infeasible_streak", 0)) + 1
        else:
            self.dwell_state["infeasible_streak"] = 0
            # Keep a known-feasible fallback instead of drifting with every switch.
            self.x_safe_default = self.x_prev

        degraded = bool(derived.shock_drift > 0.1 or derived.shock_waste_rate > 0.2)
        if degraded:
            self.dwell_state["degradation_streak"] = int(self.dwell_state.get("degradation_streak", 0)) + 1
        else:
            self.dwell_state["degradation_streak"] = 0

        anchor = self.anchor_tracker.state.anchor_cfg
        if anchor is None:
            self.seek_anchor_stable_streak = 0
        else:
            stable_radius = max(0, int(self.cfg.seek_exit_stable_radius))
            if self._switch_distance(decision.x_next, anchor) <= float(stable_radius):
                self.seek_anchor_stable_streak += 1
            else:
                self.seek_anchor_stable_streak = 0

        self.x_prev_prev = self.x_prev
        self.x_prev = decision.x_next

    def _append_history(self, window: WindowMetrics) -> None:
        row = {
            "x": self.x_prev.to_dict(),
            "z": {
                "drift": float(window.drift_value() or 0.0),
                "accept": float(window.spec_accept_mean or 0.0),
                "reverify": float(window.spec_reverify_per_step or 0.0),
                "verify_ratio": float(
                    (window.spec_verify_ms_per_step or 0.0)
                    / max(
                        (window.spec_verify_ms_per_step or 0.0) + (window.spec_draft_ms_per_step or 0.0),
                        1e-9,
                    )
                ),
                "waste_rate": float(
                    (window.rejected_tokens_per_step or 0.0)
                    / max(
                        (window.rejected_tokens_per_step or 0.0)
                        + (window.accepted_tokens_per_step or 0.0),
                        1e-9,
                    )
                ),
            },
            "y": {
                "lat_mean": float(window.lat_mean),
                "acc_mean": float(window.acc_mean),
                "train_tps": 0.0 if window.train_tps is None else float(window.train_tps),
            },
        }
        self.history.append(row)
        if len(self.history) > int(self.cfg.history_size):
            self.history = self.history[-int(self.cfg.history_size) :]
