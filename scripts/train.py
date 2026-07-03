from trackastra.training import (
    build_dataset,
    build_model,
    build_trainer,
    load_sequences,
    parse_training_config,
)

if __name__ == "__main__":
    model_config, data_train_config, data_val_config, train_config = (
        parse_training_config()
    )

    train_sequences = load_sequences(data_train_config.sources)
    val_sequences = load_sequences(data_val_config.sources)

    train_dataset = build_dataset(train_sequences, data_train_config)
    val_dataset = build_dataset(val_sequences, data_val_config)

    model = build_model(model_config, train_dataset)
    trainer = build_trainer(train_config)

    trainer.fit(model, train_dataset, val_dataset)
