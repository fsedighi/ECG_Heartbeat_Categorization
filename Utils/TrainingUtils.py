# import mlflow

from tensorflow import keras
import TransformerBasedModel as transformer_based_model
from sklearn.model_selection import GridSearchCV


def train_model(x_train, y_train, epochs=10, patience=10, learning_rate=1e-4):
    
    input_shape = x_train.shape[1:]
    
    model = transformer_based_model.build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
    learning_rate = learning_rate
)

    
    callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
    history = model.fit(
                x_train,
                y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
            )
    
    return model, history


def build_model_wrapper(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, mlp_dropout, dropout, learning_rate):
    model = transformer_based_model.build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        mlp_dropout,
        dropout,
        learning_rate=learning_rate
    )

    return model
    
def hyperparameter_tuning(x_train, y_train, hyperparameters):
    
    input_shape = x_train.shape[1:]
    # Create a wrapper function with fixed input_shape
    model = keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=build_model_wrapper,
        input_shape=input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        batch_size=64
    )

    # Create GridSearchCV object
    grid_search = GridSearchCV(model, hyperparameters)

    # Train the model with the grid search
    grid_search.fit(x_train, y_train)

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    return best_model, best_params





