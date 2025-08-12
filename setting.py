class Settings:
    def __init__(
        self,
        num_trials: int,
        num_processes: int,
        T: int,
        V_IDS_samples: int,
        num_arms: int,
        base_seed: int = 0,
    ) -> None:
        self.num_trials: int = num_trials
        self.num_processes: int = num_processes
        self.T: int = T
        self.V_IDS_samples: int = V_IDS_samples
        self.num_arms: int = num_arms
        self.base_seed = base_seed


_settings = None


def init_setting(
    num_trials: int = 100,
    num_processes: int = 10,
    T: int = 500,
    V_IDS_samples: int = 10000,
    num_arms: int = 10,
    base_seed: int = 0,
) -> None:
    global _settings
    _settings = Settings(
        num_trials, num_processes, T, V_IDS_samples, num_arms, base_seed
    )


def get_settings() -> Settings:
    if _settings is not None:
        return _settings
    raise Exception("Settings not initialized")
