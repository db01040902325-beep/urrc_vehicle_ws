"""Suppress isolated opposite-direction steering commands without blocking real turns."""


class SteeringReversalGuard:
    def __init__(self, deadband_deg=3.0, confirm_frames=3):
        self.deadband_deg = float(deadband_deg)
        self.confirm_frames = max(1, int(confirm_frames))
        self.last_sign = 0
        self.opposite_frames = 0

    def filter(self, target_deg):
        target = float(target_deg)
        if abs(target) <= self.deadband_deg:
            self.opposite_frames = 0
            return target

        sign = 1 if target > 0.0 else -1
        if self.last_sign == 0 or sign == self.last_sign:
            self.last_sign = sign
            self.opposite_frames = 0
            return target

        self.opposite_frames += 1
        if self.opposite_frames >= self.confirm_frames:
            self.last_sign = sign
            self.opposite_frames = 0
            return target

        # Neutral output while confirming a genuine direction reversal.
        return 0.0

    def reset(self):
        self.last_sign = 0
        self.opposite_frames = 0
