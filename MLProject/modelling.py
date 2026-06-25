import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import numpy as np

def main():
    print("Generating simulated data for CI/CD test...")
    x_train = np.random.rand(100, 128, 128, 1).astype('float32')
    y_train = np.random.randint(0, 25, 100)
    
    with mlflow.start_run():
        model = models.Sequential([
            layers.InputLayer(shape=(128, 128, 1)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(25, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("Training model...")
        model.fit(x_train, y_train, epochs=1, verbose=0)
        
        print("Logging model to MLflow...")
        mlflow.tensorflow.log_model(model, "model_pe_malware")
        
        print("CI Pipeline Test Completed Successfully!")

if __name__ == "__main__":
    main()