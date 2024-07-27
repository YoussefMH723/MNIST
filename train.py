import torch
import data_setup, model_builder, engine, utils

DATA_PATH = 'data/'
MODEL_NAME = "cnn_MNIST"
NUM_EPOCHS = 3
BATCH_SIZE = 64
HIDDEN_UNITS = 16
LEARNING_RATE = 0.01

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(data_path=DATA_PATH,
                                                                               batch_size=BATCH_SIZE)

print("DataLoaders are ready...")

model = model_builder.CNN(input_shape=1,
                          hidden_units=HIDDEN_UNITS,
                          output_shape=len(class_names))

print("Model is built...")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
result = engine.train(model,
                      train_dataloader,
                      test_dataloader,
                      loss_fn,
                      optimizer,
                      NUM_EPOCHS)

utils.save_model(model, saving_dir="model/", model_name=MODEL_NAME)

utils.save_training_result(result=result, model_name=MODEL_NAME)