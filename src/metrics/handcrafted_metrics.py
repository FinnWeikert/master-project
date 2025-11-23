# path length, reversals, smoothness


class HandMetrics:
    def __init__(self, hand_trajectory, velocity_threshold=20.0):
        self.df = hand_trajectory.df.copy()
        self.fps = hand_trajectory.fps
        self.hand_label = hand_trajectory.hand_label
        self.velocity_threshold = velocity_threshold

    def compute_extended_hand_metrics(self):
        df = self.df.copy().sort_values("frame")

        # --- dt ---
        df["frame_diff"] = df["frame"].diff().fillna(1)
        df["dt"] = df["frame_diff"] / self.fps

        # --- velocity & acceleration ---
        df["velocity"] = df["disp_filtered"] / df["dt"].replace(0, np.nan)
        df["accel"] = df["velocity"].diff() / df["dt"].replace(0, np.nan)

        # --- total path ---
        total_path = df["disp_filtered"].sum()
        duration = df["dt"].sum()

        # --- curvature measures ---
        dx = np.diff(df["cx_smooth"].ffill())
        dy = np.diff(df["cy_smooth"].ffill())
        path_vec = np.vstack([dx, dy]).T
        disp = df["disp_filtered"].values[1:]

        MIN_MOVE = 5
        valid = disp > MIN_MOVE
        path_vec = path_vec[valid]

        if len(path_vec) > 1:
            angles = (np.arctan2(path_vec[1:,1], path_vec[1:,0]) -
                      np.arctan2(path_vec[:-1,1], path_vec[:-1,0]))
            angles = np.mod(angles + np.pi, 2*np.pi) - np.pi
            mean_abs_angle_change = np.mean(np.abs(angles))
            num_reversals = np.sum(np.diff(np.sign(angles)) != 0)
        else:
            mean_abs_angle_change = 0
            num_reversals = 0

        # --- efficiency & temporal metrics ---
        mean_velocity = df["velocity"].mean(skipna=True)
        rms_accel = np.sqrt((df["accel"]**2).mean(skipna=True))
        efficiency = total_path / duration if duration > 0 else np.nan

        # --- duty cycle ---
        moving_time = (df["velocity"] > self.velocity_threshold).multiply(df["dt"]).sum()
        duty_cycle = moving_time / duration if duration > 0 else np.nan

        # --- tremor ---
        stationary_mask = df["velocity"] < self.velocity_threshold
        tremor_rms = np.sqrt((df.loc[stationary_mask, 'velocity']**2).mean()) \
                     if stationary_mask.any() else 0

        # --- intermittency ---
        moving = df["velocity"] > self.velocity_threshold
        intermittency_ratio = (moving.astype(int).diff().abs().sum() / len(moving)
                               if moving.any() else 0)

        metrics = pd.DataFrame([{
            f"total_path_{self.hand_label}": total_path,
            f"total_duration_{self.hand_label}": duration,
            f"mean_velocity_{self.hand_label}": mean_velocity,
            f"rms_accel_{self.hand_label}": rms_accel,
            f"efficiency_{self.hand_label}": efficiency,
            f"duty_cycle_{self.hand_label}": duty_cycle,
            f"tremor_rms_{self.hand_label}": tremor_rms,
            f"mean_abs_angle_change_{self.hand_label}": mean_abs_angle_change,
            f"num_reversals_{self.hand_label}": num_reversals,
            f"intermittency_ratio_{self.hand_label}": intermittency_ratio
        }])

        return metrics
