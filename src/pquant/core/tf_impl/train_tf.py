import keras

from pquant.core.tf_impl.compressed_layers_tf import (
    call_post_round_functions,
    post_epoch_functions,
    post_pretrain_functions,
    pre_epoch_functions,
    pre_finetune_functions,
    save_weights_functions,
)


def iterative_train_tf(model, config, train_func, valid_func, callbacks=None, **kwargs):
    """
    Generic training loop, user provides training and validation functions
    """
    callbacks = callbacks or []
    epoch = keras.ops.convert_to_tensor(0)  # Keeps track of all the epochs completed
    training_config = config["training_parameters"]
    if training_config["pretraining_epochs"] > 0:
        print("Pretraining...")
        for e in range(training_config["pretraining_epochs"]):
            pre_epoch_functions(model, e, training_config["pretraining_epochs"])
            train_loss = train_func(model, epoch=epoch, **kwargs)
            val_loss = valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["pretraining_epochs"])
            
            if any(cb.on_epoch_end(int(epoch), train_loss, val_loss, model) for cb in callbacks):
                print("Early-stopped by callback.")
                break
            epoch += 1
    
    post_pretrain_functions(model, config)
    
    print("Training...")
    for r in range(training_config["rounds"]):
        for e in range(training_config["epochs"]):
            if r == 0 and training_config["save_weights_epoch"] == e:
                save_weights_functions(model)
            pre_epoch_functions(model, e, training_config["epochs"])
            train_loss = train_func(model, epoch=epoch, **kwargs)
            val_loss = valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["epochs"])
            if any(cb.on_epoch_end(int(epoch), train_loss, val_loss, model) for cb in callbacks):
                print("Early-stopped by callback.")
                break
            epoch += 1
        call_post_round_functions(model, training_config["rewind"], training_config["rounds"], r)
    
    pre_finetune_functions(model)
    print("Fine-tuning...")
    if training_config["fine_tuning_epochs"] > 0:
        for e in range(training_config["fine_tuning_epochs"]):
            pre_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            train_loss = train_func(model, epoch=epoch, **kwargs)
            val_loss = valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            if any(cb.on_epoch_end(int(epoch), train_loss, val_loss, model) for cb in callbacks):
                print("Early-stopped by callback.")
                break
            epoch += 1
    return model
