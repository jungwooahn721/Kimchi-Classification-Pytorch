import matplotlib.pyplot as plt

classwise_accuracy = {0: 0.285, 1: 0.785, 2: 0.36, 3: 0.455, 4: 0.375, 5: 0.59, 6: 0.27, 7: 0.42, 8: 0.395, 9: 0.385, 10: 0.375}

# Convert the dictionary into two lists: one for class indices and one for accuracy values
class_indices, accuracies = zip(*classwise_accuracy.items())

# Plot the bar chart
plt.bar(class_indices, accuracies, color='blue')
plt.xlabel('Class Index')
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')
plt.ylim(0, 1)  
plt.show()
