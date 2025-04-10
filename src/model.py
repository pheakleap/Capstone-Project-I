import pandas as pd
import tensorflow as tf

# Load and preprocess data
def load_data():
    training_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    training_df['median_house_value'] /= 1000.0
    return training_df

# Build the model
def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Train the model
def train_model(model, df, feature, label, epochs, batch_size):
    history = model.fit(x=df[feature], y=df[label], batch_size=batch_size, epochs=epochs)
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse

# Predict function
def predict(model, input_data):
    return model.predict(input_data)
