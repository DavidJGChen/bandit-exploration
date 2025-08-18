from dataclasses import dataclass


@dataclass
class Settings:
    num_trials: int
    num_processes: int
    T: int
    num_arms: int
    base_seed: int
    multiprocessing: bool
    trial_id_overrides: list[int] | None
    output_dir: str


_settings: Settings | None = None


def init_setting(
    num_trials: int = 100,
    num_processes: int = 10,
    T: int = 500,
    num_arms: int = 10,
    base_seed: int = 0,
    multiprocessing: bool = True,
    trial_id_overrides: list[int] | None = None,
    output_dir: str = "output/temp/",
) -> None:
    global _settings
    _settings = Settings(
        num_trials,
        num_processes,
        T,
        num_arms,
        base_seed,
        multiprocessing,
        trial_id_overrides,
        output_dir,
    )


def get_settings() -> Settings:
    if _settings is not None:
        return _settings
    raise Exception("Settings not initialized")


def generate_base_filename(base_seed: int, trial_id: int) -> str:
    return f"seed{base_seed}_id{trial_id}.parquet"
