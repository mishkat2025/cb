"""Plain-language labels for the Streamlit app (demo/app.py): exercise, camera and recording-condition names, the
Module III error types, angle channels, per-rep measures, the EAAQ items and the model names.

Display only: nothing here changes a number. Meanings come from MobiPhysio.md, configs/exercises/EXX.yaml,
cvdc/assess/module3.py and DESIGN_DECISIONS.md (D5, D11, D13, M11, M12).
"""

EXERCISES = {"E01": "Shoulder abduction", "E02": "Shoulder adduction", "E03": "Shoulder lateral rotation",
             "E04": "Shoulder medial rotation", "E05": "Shoulder circumduction", "E06": "Wrist extension",
             "E07": "Hip joint flexion", "E08": "Both-leg flexion (lumbar)", "E09": "Back extension"}

CAMERAS = {"F": "Front", "L": "Left side", "R": "Right side"}

CONDITIONS = {"FL": "Full light", "ML": "Medium light", "LL": "Low light", "LJ": "Slight camera shake",
              "HJ": "Strong camera shake", "O": "Body partly hidden", "LR": "Low resolution", "unknown": "Other"}

GROUPS = {"expert": "Expert", "non-expert": "Non-expert"}
GENDERS = {"M": "Male", "F": "Female"}

QUALITY = {"correct": "Correct", "partially_correct": "Partly correct", "incorrect": "Incorrect"}
QUALITY_RULE = "Correct = 80 or more, Partly correct = 50 to 80, Incorrect = below 50."

SEVERITY = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}
SEVERITY_RULE = ("How far the movement went outside the range the high-scoring experts stay in: Mild = up to 1 expert "
                 "SD beyond it, Moderate = 1 to 2 SD, Severe = more than 2 SD.")

NO_MODEL = "none (Module III only)"
MODELS = {
    "proposed_j1": ("Proposed model (default)",
                    "Learned that the same rep filmed by different cameras is the same movement, then trained on "
                    "the physiotherapists' scores."),
    "b1": ("Baseline model (B1)", "The same network trained from scratch on 2D keypoints, without the "
                                   "cross-camera step. For comparison."),
    "proposed_lp": ("Proposed model, frozen", "The cross-camera network kept fixed; only the small scoring layer was "
                                               "trained."),
    NO_MODEL: ("No model: rule-based check only", "Only Module III (each rep compared with the physiotherapist's "
                                                  "demo). Needs no trained model."),
}

# Module III angle channels (configs/exercises/EXX.yaml, cvdc/assess/angles.py), all in degrees
CHANNELS = {
    "shoulder_elevation": "Arm raise (upper arm vs trunk)",
    "horizontal_adduction": "Arm across the body",
    "elbow_extension": "Elbow straightness (180 = straight)",
    "contralateral_elevation": "Other arm raise",
    "trunk_lateral_lean": "Trunk side lean",
    "shoulder_asymmetry": "Uneven shoulders (shrug)",
    "forearm_elevation": "Forearm angle (rotation)",
    "wrist_extension": "Wrist bend",
    "hip_flexion": "Hip bend",
    "knee_flexion": "Knee bend",
    "contralateral_hip_flexion": "Other hip bend",
    "contralateral_knee_flexion": "Other knee bend",
    "trunk_thigh_angle": "Trunk-thigh angle",
    "knee_flexion_both": "Knee bend (both legs)",
    "hip_flexion_asymmetry": "Hip difference, left vs right",
    "knee_flexion_asymmetry": "Knee difference, left vs right",
}

# the main movement named in each "<movement>_range_insufficient" error type
MOVEMENTS = {"abduction": "arm raise to the side", "adduction": "arm movement across the body",
             "lateral_rotation": "outward rotation", "medial_rotation": "inward rotation",
             "circumduction": "arm circle", "wrist_extension": "wrist bend", "hip_flexion": "hip bend",
             "trunk_extension": "back extension"}

ERRORS = {
    "elbow_bent": ("Elbow bent", "The elbow should stay straight."),
    "elbow_angle_deviation": ("Elbow angle off", "The elbow angle is different from the target."),
    "other_arm_raised": ("Other arm raised", "The arm that should rest by the side moved up."),
    "trunk_lean": ("Trunk leaning sideways", "The body tilts to one side instead of staying upright."),
    "shoulder_asymmetry": ("Shoulder hitched", "One shoulder is pulled up (shrugged) compared with the other."),
    "shoulder_hitch": ("Shoulder hitched", "One shoulder is pulled up (shrugged) compared with the other."),
    "arm_elevation_deviation": ("Arm height off", "The arm is held higher or lower than the target."),
    "shoulder_abduction_deviation": ("Upper-arm position off", "The upper arm is lifted away from the body more or "
                                                               "less than the target."),
    "shoulder_elevation_deviation": ("Upper-arm position off", "The upper arm is raised more or less than the target."),
    "forearm_position_deviation": ("Forearm position off", "The forearm is held higher or lower than the target."),
    "knee_angle_deviation": ("Knee angle off", "The knee bends more or less than the target."),
    "knee_flexion_deviation": ("Knee bend off", "The knee bends more or less than the target."),
    "other_hip_flexed": ("Other hip bent", "The leg that should stay still bends at the hip."),
    "other_knee_flexed": ("Other knee bent", "The leg that should stay still bends at the knee."),
    "hip_asymmetry": ("Hips uneven", "The left and right hips move differently."),
    "knee_asymmetry": ("Knees uneven", "The left and right knees move differently."),
    "knees_flexed": ("Knees bent", "The knees should stay straight."),
    "too_fast": ("Too fast", "This repetition was quicker than the experts do it."),
    "too_slow": ("Too slow", "This repetition was slower than the experts do it."),
    "too_few_reps": ("Too few repetitions", "The set has fewer repetitions than the exercise asks for."),
}

# EAAQ items (Physiotherapist Exercise Marking.csv: 4 primary factors, 3 control factors; PROGRESS.md §3, D5, D11)
ITEMS = {
    "PF1": "Primary factor 1",
    "PF2": "Primary factor 2",
    "PF3": "Primary factor 3 (repetitions done, yes/no)",
    "PF4": "Primary factor 4 (repetitions, at least 10)",
    "CF1": "Control factor 1 (posture / control)",
    "CF2": "Control factor 2 (posture / control)",
    "CF3": "Control factor 3 (posture / control)",
}
ITEMS_NOTE = ("The physiotherapists filled in a 7-question form (EAAQ) per video: 4 primary factors (speed and "
              "repetitions) and 3 control factors (posture and movement quality), each 0 to 1. The exact wording of "
              "each question is not in the dataset yet.")

MEASURES = {
    "rom_deg": ("Range reached", "deg", "How far the main angle went in this repetition."),
    "rom_target_deg": ("Target range", "deg", "The range the reference expects for this camera view."),
    "rom_shortfall_deg": ("Short of the target by", "deg", "Target range minus range reached (negative = went "
                                                           "further than the target)."),
    "excursion_deg": ("Total movement of the main angle", "deg", "Largest minus smallest value of the main angle "
                                                                  "in the repetition."),
    "duration_s": ("Duration", "s", "Time from the start to the end of the repetition."),
    "speed_ratio": ("Speed compared with the PT demo", "x", "Demo duration / this repetition's duration: above 1 = "
                                                             "faster than the physiotherapist, below 1 = slower."),
    "angular_velocity_dev_deg_s": ("Speed-profile difference", "deg/s", "Average difference between this "
                                                                         "repetition's angular speed and the "
                                                                         "target's, moment by moment."),
}

SCORE_PARTS = {"angle": "Posture angles", "rom": "Range of motion", "speed": "Speed", "symmetry": "Left-right symmetry"}

PANELS = ("**Top left:** the patient video with the detected skeleton (faces blurred). Joints with a problem in the "
          "last repetition turn red. Labels at the joints show the patient's angle minus the target, green inside "
          "the allowed range and red outside it.\n\n"
          "**Top right:** the physiotherapist's own demo skeleton, moving in step with the patient's repetition.\n\n"
          "**Bottom left:** an estimated 3D view (single camera) of the patient (green) and the physiotherapist "
          "(cyan), slowly turning.\n\n"
          "**Bottom right:** the model's view. Each finished repetition is a numbered point; the cyan star is the "
          "physiotherapist's demo. Points closer to the star look more like the demo to the model. The faint "
          "cloud is the training repetitions, coloured by their score; the gauge shows the latest repetition's "
          "distance to the demo.")


def label(d, key):
    """d[key], or the key itself made readable."""
    return d.get(key) or str(key).replace("_", " ").capitalize()


def exercise(ex):
    return f"{ex} · {EXERCISES.get(ex, ex)}"


def joint(name):
    return str(name or "").replace("_", " ")


def error_title(e):
    t = e.get("type", "")
    if t.endswith("_range_insufficient"):
        return f"Movement too small ({MOVEMENTS.get(t[:-len('_range_insufficient')], 'main movement')})"
    return label({k: v[0] for k, v in ERRORS.items()}, t)


def error_sentence(e, fps=30.0):
    """One plain sentence for a Module III error (rep-level or set-level)."""
    t = e.get("type", "")
    if t == "too_few_reps":
        return (f"{e.get('measured_reps')} repetitions found; the exercise asks for at least "
                f"{e.get('target_reps')} ({abs(e.get('deviation_reps', 0))} missing).")
    where = f" ({joint(e['joint'])})" if e.get("joint") else ""
    when = ""
    if e.get("frames"):
        a, b = (x / fps for x in e["frames"])
        when = f" At {a:.1f} s." if abs(b - a) < 0.05 else f" From {a:.1f} s to {b:.1f} s."
    if "measured_s" in e:  # speed rules
        return (f"This repetition took {e['measured_s']:.2f} s; experts take about {e['target_s']:.2f} s "
                f"({abs(e['deviation_s']):.2f} s {'shorter' if e['deviation_s'] < 0 else 'longer'}).{when}")
    m, tg, d = e.get("measured_deg"), e.get("target_deg"), e.get("deviation_deg")
    if t.endswith("_range_insufficient"):
        return (f"The {MOVEMENTS.get(t[:-len('_range_insufficient')], 'main movement')}{where} reached {m:.0f}° "
                f"but the target is {tg:.0f}°, so it fell {abs(d):.0f}° short.{when}")
    intro = ERRORS.get(t, ("", ""))[1]
    if m is None or tg is None or d is None:
        return f"{intro}{where}{when}".strip()
    subject = f"The {joint(e['joint'])} was at" if e.get("joint") else "Measured"
    return (f"{intro} {subject} {m:.0f}° against a target of {tg:.0f}° ({abs(d):.0f}° "
            f"{'more' if d > 0 else 'less'}).{when}").strip()
