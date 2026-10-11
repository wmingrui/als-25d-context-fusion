def install_fixed_eval_seed(trainer, seed=42):
    original = trainer.stable_seed
    def fixed_seed(training_seed, name):
        return original(seed, name)
    trainer.stable_seed = fixed_seed
    return original
