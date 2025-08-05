class Settings:
    def __init__(self,
                 num_trials: int,
                 num_processes : int,
                 T: int,
                 V_IDS_samples: int,
                 num_arms: int) -> None:
        self.num_trials: int = num_trials
        self.num_processes: int = num_processes
        self.T: int = T
        self.V_IDS_samples: int = V_IDS_samples
        self.num_arms: int = num_arms


_settings = None


def init_setting(
        num_trials: int = 100,
        num_processes: int = 10,
        T: int = 500,
        V_IDS_samples: int = 10000,
        num_arms: int = 10
) -> None:
    global _settings
    _settings = Settings(num_trials, num_processes, T, V_IDS_samples, num_arms)


def get_settings() -> Settings:
    return _settings
