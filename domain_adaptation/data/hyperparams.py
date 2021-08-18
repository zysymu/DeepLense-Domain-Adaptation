def supervised_hyperparams(lr=1e-3, wd=1e-5, scheduler=True):
    hyperparams = {'learning_rate': lr,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams

def adda_hyperparams(lr_target=1e-6, lr_discriminator=1e-5, wd=1e-5, scheduler=True):
    hyperparams = {'learning_rate_target': lr_target,
                   'learning_rate_discriminator': lr_discriminator,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams

def self_ensemble_hyperparams(lr=1e-4, unsupervised_weight=3.0, wd=1e-5, scheduler=True):
    hyperparams = { 'learning_rate': lr,
                   'unsupervised_weight': unsupervised_weight,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams

def cgdm_hyperparams():
    pass

def adamatch_hyperparams(lr=1e-4, wd=1e-5, scheduler=True, tau=0.9):
    hyperparams = {'learning_rate': lr,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler,
                   'tau': tau
                   }

    return hyperparams