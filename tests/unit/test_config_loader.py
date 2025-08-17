from fbm.config.loader import load_config

def test_load_config_reads_yaml_defaults():
    cfg = load_config("conf/default.yaml")
    assert "paths" in cfg and "datalake" in cfg["paths"]
    assert "betting" in cfg and "bankroll" in cfg["betting"]
    assert isinstance(cfg["betting"]["bankroll"], (int, float))
