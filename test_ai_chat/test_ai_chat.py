import numpy as np
import tensorflow as tf

class SimpleAI:
    def __init__(self):
        self.model = self.build_model()
        self.history = []
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, inputs, outputs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        self.model.fit(inputs, outputs, epochs=100, verbose=0)
    
    def predict(self, input_data):
        input_data = np.array(input_data).reshape(-1, 1)
        return self.model.predict(input_data)[0][0]

def main():
    ai = SimpleAI()
    print("Hello! I'm your artificial intelligence.")
    print("I can learn from your examples.")
    
    while True:
        user_input = input("Your input (1 for positive example, 0 for negative example): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            user_input = int(user_input)
            if user_input == 1 or user_input == 0:
                ai.history.append(user_input)
            else:
                print("Please enter only 1 or 0.")
                continue
            
            inputs = np.arange(len(ai.history)).reshape(-1, 1)
            ai.train(inputs, ai.history)
            print("Model trained.")
        except ValueError:
            print("Please enter only 1 or 0.")
            continue

if __name__ == "__main__":
    main()

