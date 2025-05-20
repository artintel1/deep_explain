# Script to implement and demonstrate Integrated Gradients

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_dataset():
    # Load MNIST dataset from Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data:
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print("MNIST dataset loaded and preprocessed.")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def get_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(x_train, y_train, x_test, y_test, model_path='mnist_cnn_model.keras'):
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]
    model = get_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Training the model...")
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5, # Reduced for speed
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model

def integrated_gradients(model, input_image, target_class_index, baseline, m_steps=100):
    input_image = tf.convert_to_tensor(input_image)
    baseline = tf.convert_to_tensor(baseline)
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    gradient_sum = tf.zeros_like(input_image, dtype=tf.float32)

    for i in range(m_steps + 1):
        alpha = alphas[i]
        interpolated_image = baseline + alpha * (input_image - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_image)
            predictions = model(interpolated_image, training=False)
            target_output = predictions[:, target_class_index]
        gradients = tape.gradient(target_output, interpolated_image)
        if gradients is not None:
            gradient_sum += gradients
        else:
            print(f"Warning: Gradients are None for alpha = {alpha.numpy()}. Skipping.")
    
    if m_steps > 0:
        avg_gradients = gradient_sum / tf.cast(m_steps, tf.float32)
    else:
        avg_gradients = gradient_sum 
    attributions = (input_image - baseline) * avg_gradients
    return attributions.numpy()

def visualize_attributions(original_image, attributions, title="Integrated Gradients Attributions"):
    if original_image.ndim == 4 and original_image.shape[0] == 1:
        original_image_plot = np.squeeze(original_image, axis=0)
    else:
        original_image_plot = original_image
    if attributions.ndim == 4 and attributions.shape[0] == 1:
        attributions_plot = np.squeeze(attributions, axis=0)
    else:
        attributions_plot = attributions
    if original_image_plot.ndim == 3 and original_image_plot.shape[-1] == 1:
        original_image_plot = np.squeeze(original_image_plot, axis=-1)
    if attributions_plot.ndim == 3 and attributions_plot.shape[-1] == 1:
        attributions_plot = np.squeeze(attributions_plot, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image_plot, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    abs_max = np.abs(attributions_plot).max()
    if abs_max == 0: abs_max = 1.0
    im = axes[1].imshow(attributions_plot, cmap='bwr', vmin=-abs_max, vmax=abs_max)
    axes[1].set_title(title)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path = "integrated_gradients_visualization.png"
    plt.savefig(save_path)
    print(f"Attribution visualization saved to {save_path}")
    plt.close(fig)

def run_mnist_integrated_gradients_example():
    """
    Runs a demonstration of Integrated Gradients on the MNIST dataset.
    This includes:
    1. Loading and preprocessing the MNIST dataset.
    2. Loading a pre-trained CNN model or training a new one.
    3. Selecting a sample image from the test set.
    4. Computing Integrated Gradients for the model's prediction on the sample image.
    5. Visualizing the original image and its attributions, then saving the plot.
    """
    print("Starting MNIST Integrated Gradients example...")
    print("This script will load/train a CNN, pick an MNIST test image,")
    print("compute Integrated Gradients for that image, and save the resulting")
    print("visualization to 'integrated_gradients_visualization.png'.")

    # Step 1: Load and preprocess the MNIST dataset
    # This function handles loading, normalization, and reshaping of MNIST data.
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_dataset()

    # Step 2: Load a pre-trained model or train a new one
    # The model is a simple CNN for MNIST classification.
    # If a saved model ('mnist_cnn_model.keras') exists, it's loaded; otherwise,
    # the script trains a new model and saves it.
    model_path = 'mnist_cnn_model.keras'
    try:
        trained_model = tf.keras.models.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
    except IOError:
        print(f"Model not found at {model_path}. Training a new model...")
        trained_model = train_model(x_train, y_train, x_test, y_test, model_path)

    # Step 3: Select a sample image and define a baseline
    # We'll use the first image from the test set for this example.
    # The baseline is a black image (all zeros), a common choice for IG.
    sample_img_index = 0 
    sample_image = x_test[sample_img_index:sample_img_index+1].astype('float32') # Ensure batch dim and float32
    sample_label_one_hot = y_test[sample_img_index:sample_img_index+1]
    sample_label_index = np.argmax(sample_label_one_hot) # True label
    baseline_image = np.zeros_like(sample_image, dtype=np.float32)

    # Step 4: Make a prediction and print information
    # This helps confirm the model's prediction for the chosen sample.
    print(f"\nExplaining prediction for image at index {sample_img_index} from the test set.")
    print(f"True label of the image: {sample_label_index}")
    
    predictions = trained_model(sample_image, training=False) # Get model's prediction
    predicted_class_index = np.argmax(predictions[0])
    prediction_confidence = predictions[0][predicted_class_index]
    print(f"Model's predicted class: {predicted_class_index} with confidence: {prediction_confidence:.4f}")

    # Step 5: Compute Integrated Gradients
    # This is the core XAI technique. We compute attributions for the predicted class.
    # m_steps defines the number of steps in the integration approximation.
    print("\nComputing Integrated Gradients...")
    attributions = integrated_gradients(
        trained_model, 
        sample_image, 
        predicted_class_index, 
        baseline_image, 
        m_steps=100  # Number of steps for the approximation
    )
    print(f"Integrated Gradients computed. Attributions array shape: {attributions.shape}")
    print(f"Attributions min: {np.min(attributions):.4f}, max: {np.max(attributions):.4f}, mean: {np.mean(attributions):.4f}")

    # Step 6: Visualize the attributions
    # This function plots the original image alongside its attribution map
    # and saves it to a file.
    print("\nVisualizing attributions and saving the plot...")
    visualization_title = (f"Integrated Gradients for Predicted Class: {predicted_class_index} "
                           f"(True Class: {sample_label_index})")
    visualize_attributions(sample_image, attributions, title=visualization_title)
    
    print("\nMNIST Integrated Gradients example finished.")
    print("Check 'integrated_gradients_visualization.png' for the output.")

if __name__ == '__main__':
    run_mnist_integrated_gradients_example()
