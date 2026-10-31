from dataclasses import dataclass


@dataclass
class AppConfig:
    # Camera indices. On a MacBook, built-in camera is often 0; external USB camera may be 1 or 2.
    front_camera_index: int = 0
    side_camera_index: int = 1
    # Capture mode request (0 x 0 = camera default). 4:3 uses the full sensor height: measured on the
    # MacBook FaceTime HD camera (+23% vertical view) and iPhone Continuity Camera (+39% vertical, +7%
    # horizontal) versus their 16:9 modes, which crop the top and bottom. Cameras pick their nearest mode.
    frame_width: int = 1280
    frame_height: int = 960
    target_fps: int = 20
    processing_max_width: int = 1280  # bigger frames are downscaled (aspect kept, never cropped) for speed
    camera_frame_timeout_seconds: float = 2.0  # no new frame for this long -> camera error

    # Pose model
    pose_model_complexity: int = 1           # 0 lite, 1 full, 2 heavy (more accurate, ~2-3x slower)
    mediapipe_smooth_landmarks: bool = True  # MediaPipe's built-in temporal filter
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55

    # Landmark gating: a landmark is only drawn/measured while its confidence (visibility, and
    # presence when MediaPipe provides it) is above the threshold. Once shown it may drop by
    # `hysteresis` before being hidden, which stops flicker at the boundary.
    landmark_visibility_threshold: float = 0.5        # front camera
    side_landmark_visibility_threshold: float = 0.5   # side camera (far-side landmarks are often guessed)
    landmark_visibility_hysteresis: float = 0.1
    landmark_edge_margin: float = 0.02                # landmarks this far outside the frame are rejected
    landmark_max_jump_ratio: float = 0.15             # max move per frame, as a fraction of the frame diagonal
    landmark_outlier_max_frames: int = 3              # a jump that persists longer than this is accepted

    # One Euro filter on landmark pixel positions (lower min_cutoff = less jitter, higher beta = less lag)
    landmark_filter_min_cutoff: float = 1.2
    landmark_filter_beta: float = 0.01
    landmark_filter_d_cutoff: float = 1.0

    # Side camera: keep tracking the same body side unless the other is clearly more visible
    side_switch_margin: float = 0.15
    # Seated hips are often partly hidden by the thigh/desk/armrest; MediaPipe still places them well,
    # so the side camera accepts hips at a lower confidence than other landmarks.
    side_hip_visibility_threshold: float = 0.35

    # Feature smoothing (landmarks are already One-Euro filtered, so this can stay fairly responsive)
    ema_alpha: float = 0.35
    feature_reset_after_missing_frames: int = 10  # don't blend stale values after the pose was lost
    measurement_hold_seconds: float = 0.5  # keep the last reliable reading through brief landmark drop-outs

    # Warning debounce: an indicator must stay outside its limit this long before it is flagged, and back
    # below limit * hysteresis_ratio this long before it clears (ignores brief movements, stops flicker).
    metric_enter_seconds: float = 1.5
    metric_exit_seconds: float = 1.0
    metric_hysteresis_ratio: float = 0.85
    # Severity (magnitude AND duration). A flagged indicator is WARNING. It becomes POOR when it is severe
    # (deviation > metric_poor_ratio x its limit) for poor_severe_seconds, or when a not-minor deviation
    # (>= poor_persistent_min_ratio x limit) has lasted poor_posture_seconds. Minor deviations stay WARNING.
    # metric_poor_ratio also sets the score bands: indicator score 80 at 1x the limit, 60 at 1.5x.
    metric_poor_ratio: float = 1.5
    poor_severe_seconds: float = 5.0
    poor_persistent_min_ratio: float = 1.25
    poor_posture_seconds: float = 20.0

    # Popup notifications (the only notification UI; see notifications.py for the episode rules)
    popup_trigger_seconds: float = 10.0   # confirmed bad posture (reliable data only) before the warning popup
    popup_visible_seconds: float = 15.0   # a popup closes itself after being visible this long
    popup_recovery_seconds: float = 3.0   # continuous GOOD posture that closes the popup and ends the episode
    popup_snooze_seconds: float = 60.0
    popup_inactive_reset_seconds: float = 60.0  # no posture evidence this long (user away) ends the episode

    # ---------------------------------------------------------------------------------------------
    # Ergonomic reference limits (no personal calibration). See README "Posture metrics" for formulas.
    # Published guidance used as the basis:
    #   * OSHA eTool "Computer Workstations": neutral upright sitting; reclined sitting with the torso
    #     105-120 deg from the thighs (i.e. up to ~30 deg behind vertical) is an acceptable posture;
    #     monitor 15-20 deg below horizontal eye level.
    #   * RULA (McAtamney & Corlett 1993): neck flexion bands 0-10 / 10-20 / >20 deg; trunk flexion
    #     bands 0-20 / 20-60 deg; any neck or trunk side-bending adds risk.
    #   * Plumb-line alignment (Kendall et al.): ear roughly over the shoulder over the hip.
    # The numbers below are EXPERIMENTAL ENGINEERING DEFAULTS chosen from that guidance. They are not
    # OSHA limits and not clinically validated thresholds - adjust them after testing.
    shoulder_tilt_max_deg: float = 5.0          # front: shoulder line vs horizontal (experimental)
    head_tilt_max_deg: float = 10.0             # front: eye/ear line vs horizontal (experimental)
    side_lean_max_deg: float = 10.0             # front: hip-centre -> shoulder-centre vs vertical (experimental)
    upper_body_alignment_max_deg: float = 10.0  # front: shoulder-centre -> head-centre vs vertical (experimental)
    forward_head_max_deg: float = 25.0          # side: neck (shoulder -> ear) vs TRUNK axis (hip -> shoulder); RULA 20 + 5
    neck_flexion_max_deg: float = 20.0          # side: head pitched down = ear -> eye line below horizontal (RULA neck band)
    head_pitch_neutral_deg: float = 0.0         # side: ear -> eye approximates the Frankfort plane (level when the head is
                                                #   upright), so no fixed anatomical offset is assumed; adjust only after
                                                #   measuring a level head with your camera setup
    torso_forward_max_deg: float = 20.0         # side: hip -> shoulder vs vertical, forward (RULA trunk band)
    torso_recline_max_deg: float = 30.0         # side: backward recline still treated as supported (OSHA reclined sitting)
    slouch_combined_ratio: float = 0.7          # side: hunched = trunk AND forward head both above 70% of limits (heuristic)

    # Automatic camera-placement checks (experimental)
    front_min_shoulder_width_ratio: float = 0.10  # shoulders narrower than this share of the frame: too far away
    front_min_shoulder_head_ratio: float = 1.6    # shoulder width / ear distance below this: body turned away
    front_max_head_yaw_ratio: float = 0.35        # nose offset from ear centre / ear distance: head turned, skip head tilt
    side_min_torso_height_ratio: float = 0.15     # torso shorter than this share of the frame: too far away
    side_oblique_warn_ratio: float = 0.30         # shoulder x-separation / torso length: camera not side-on (warn)
    side_oblique_max_ratio: float = 0.55          # ...beyond this the side view is too oblique to measure

    # Score weights (per component; components without data are left out, never counted as perfect)
    neck_weight: float = 0.30
    back_weight: float = 0.30
    shoulder_weight: float = 0.20
    alignment_weight: float = 0.20

    # Session statistics: per-frame time steps longer than this (stalls) are not attributed to any state
    session_max_tick_seconds: float = 0.5

    # Break reminder
    break_reminder_minutes: float = 50.0

    # Session history (local SQLite database, see history.py; numbers only, never images)
    history_autosave_seconds: float = 15.0       # the open session is saved this often while the app runs
    history_min_monitored_seconds: float = 10.0  # sessions with less valid monitoring are not kept in history

    # Active Break game (fixed order: stretch -> coin -> stretch -> coin ... until the total ends)
    stretch_duration_seconds: float = 30.0
    coin_game_duration_seconds: float = 30.0
    active_break_total_seconds: float = 120.0
    active_break_transition_seconds: float = 2.5
    active_break_results_seconds: float = 20.0  # auto-return to monitoring after the results screen
    stretch_hold_seconds: float = 1.5
    stretch_exercise_timeout_seconds: float = 12.0  # skip an exercise nobody can complete
    stretch_points: int = 10
    coin_points: int = 5
    bomb_avoid_points: int = 3
    bomb_hit_penalty: int = 5

    # UI
    refresh_ms: int = 33
    mirror_front_view: bool = True
    mirror_side_view: bool = False


CONFIG = AppConfig()
