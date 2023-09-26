from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Get the true labels from the test data generator
true_labels = test_generator.classes

# Generate predictions for the test data
predictions = model.predict(test_generator)

predicted_labels = np.argmax(predictions, axis=1)
confusion_mat = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(confusion_mat)

classification_rep = classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys())

# Print the classification report
print("\nClassification Report:")
print(classification_rep)
