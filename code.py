from weka.core.jvm import start, stop
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.serialization import write

# Start Weka JVM
start()

# Load dataset
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("ARRF_DATA.arff")
data.class_is_last()

# Split dataset (80% training, 20% testing)
split_filter = Filter(
    classname="weka.filters.supervised.instance.StratifiedRemoveFolds"
)
split_filter.options = ["-N", "5", "-F", "1", "-S", "42"]  # 80% for training
split_filter.inputformat(data)
training_set = split_filter.filter(data)

split_filter.options = ["-N", "5", "-F", "2", "-S", "42"]  # 20% for testing
testing_set = split_filter.filter(data)

# Train the model
classifier = Classifier(
    classname="weka.classifiers.bayes.NaiveBayes"
)  # Change algorithm if needed
classifier.build_classifier(training_set)

# Evaluate the model
evaluation = Evaluation(training_set)
evaluation.test_model(classifier, testing_set)

# Print results
print("Summary:")
print(evaluation.summary())
print("\nClass Details:")
print(evaluation.class_details())
print("\nConfusion Matrix:")
print(evaluation.matrix())

# Save the trained model
write("trained_model.model", classifier)
print("Model saved to 'trained_model.model'")

# Stop Weka JVM
stop()
