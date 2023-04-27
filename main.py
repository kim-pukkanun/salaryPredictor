import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats

# Functions for buttons
def browse_csv():
    global dataset
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        dataset = pd.read_csv(filepath)
        input_csv_var.set(filepath)

def prediction_interval(input_data, y_pred, confidence=0.95):
    n = x_train.shape[0]
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    t_score = stats.t.ppf((1 + confidence) / 2, n - 1)
    standard_error = x_std / np.sqrt(n)
    
    margin_of_error = t_score * standard_error * np.sqrt(1 + (1 / n) + ((input_data - x_mean)**2) / np.sum((x_train - x_mean)**2))
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error
    
    return lower_bound[0][0], upper_bound[0][0]

def start_prediction():
    global x, y, x_train, x_test, y_train, y_test, regressor, y_pred

    if not dataset.empty:  # Check if dataset is loaded
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    years_of_experience = input_integer_var.get()
    if years_of_experience:
        input_data = np.array([years_of_experience]).reshape(-1, 1)
        y_pred = regressor.predict(input_data)
        
        lower_bound, upper_bound = prediction_interval(input_data, y_pred)  # Calculate prediction interval
    else:
        y_pred = regressor.predict(x_test)

    output_entry.delete(1.0, tk.END)  # Clear the output_entry widget
    if years_of_experience:
        output_entry.insert(tk.END, f"Predicted salary: {y_pred[0]:.2f}\nPrediction interval: ({lower_bound:.2f}, {upper_bound:.2f})")  # Insert the predicted salary and prediction interval
    else:
        output_entry.insert(tk.END, str(y_pred))  # Insert the predicted salary

    # Save input and output to the history file
    with open("history.txt", "a") as history_file:
        history_file.write(f"Years of experience: {years_of_experience}\n")
        history_file.write(f"Predicted salary: {y_pred[0]:.2f}\n")
        if years_of_experience:
            history_file.write(f"Prediction interval: ({lower_bound:.2f}, {upper_bound:.2f})\n\n")
        else:
            history_file.write("\n")

def pre_train():
    global dataset, x, y, x_train, x_test, y_train, y_test, regressor
    dataset = pd.read_csv("Salary_Data.csv")
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

def calculate_test_accuracy():
    y_pred_test = regressor.predict(x_test)
    test_accuracy = regressor.score(x_test, y_test)  # Calculate R-squared
    return test_accuracy * 100  # Return test accuracy as a percentage

# Pre-train the model
pre_train()

# Calculate and display test accuracy for the pre-trained model
pre_train_test_accuracy = calculate_test_accuracy()
print(f"Pre-trained model test accuracy: {pre_train_test_accuracy:.2f}%")

def train_model():
    global x, y, x_train, x_test, y_train, y_test, regressor

    if dataset.empty:
        messagebox.showerror("No Data", "Please browse and select a CSV file before training the model.")
        return

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    regressor.fit(x_train, y_train)

    # Calculate and display test accuracy for the user-trained model
    user_train_test_accuracy = calculate_test_accuracy()
    messagebox.showinfo("Model Trained", f"The model has been trained with the new data.\nTest accuracy: {user_train_test_accuracy:.2f}%")

def export_text_file():
    global y_pred
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return

    with open(file_path, "w") as text_file:
        text_file.write("Years of Experience, Predicted Salary\n")
        for experience, salary in zip(x_test.flatten(), y_pred):
            text_file.write(f"{experience}, {salary:.2f}\n")

    messagebox.showinfo("Export Successful", f"Data saved to {file_path}")

def export_csv_file():
    global y_pred
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    export_data = pd.DataFrame({"Years of Experience": x_test.flatten(), "Predicted Salary": y_pred})
    export_data.to_csv(file_path, index=False)

    messagebox.showinfo("Export Successful", f"Data saved to {file_path}")

def show_graph():
    # Create the figure and axes
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Plot the data
    ax.scatter(x_train, y_train, color="red", label="Training Set")
    ax.plot(x_train, regressor.predict(x_train), color="blue", label="Regression Line")
    ax.scatter(x_test, y_test, color="green", label="Test Set")
    ax.set_title("Salary Prediction")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()

    # Create the new window
    graph_window = tk.Toplevel(root)
    graph_window.title("Salary Prediction Graph")
    graph_window.geometry("600x500")

    # Create a canvas and draw the figure on it
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add a close button
    close_button = tk.Button(graph_window, text="Close", command=graph_window.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)

def show_history():
    with open("history.txt", "r") as history_file:
        history_data = history_file.read()

    history_window = tk.Toplevel(root)
    history_window.title("Prediction History")
    history_window.geometry("400x400")

    history_label = tk.Label(history_window, text="Prediction History:")
    history_label.pack(pady=10)

    history_text = tk.Text(history_window, wrap=tk.WORD)
    history_text.insert(tk.END, history_data)
    history_text.pack(pady=10)

    close_button = tk.Button(history_window, text="Close", command=history_window.destroy)
    close_button.pack(pady=10)

# Create the main window
root = tk.Tk()
root.title("Salary Predictor")
root.geometry("1000x500")

# Variables to store inputs and output
input_integer_var = tk.IntVar()
input_csv_var = tk.StringVar()
output_var = tk.StringVar()

# Create widgets
integer_label = tk.Label(root, text="Enter years of experience:")
integer_entry = tk.Entry(root, textvariable=input_integer_var)

csv_label = tk.Label(root, text="Select CSV file:")
csv_entry = tk.Entry(root, textvariable=input_csv_var)
csv_button = tk.Button(root, text="Browse", command=browse_csv)

start_button = tk.Button(root, text="Start Prediction", command=start_prediction)
train_button = tk.Button(root, text="Train Model", command=train_model)
export_txt_button = tk.Button(root, text="Export to Text File", command=export_text_file)
export_csv_button = tk.Button(root, text="Export to CSV File", command=export_csv_file)
graph_button = tk.Button(root, text="Show Graph", command=show_graph)
history_button = tk.Button(root, text="Show History", command=show_history)

output_label = tk.Label(root, text="Predicted Salary:")
output_entry = tk.Text(root, height=10, width=100, wrap=tk.WORD)

# Position widgets
integer_label.grid(row=0, column=0, pady=10)
integer_entry.grid(row=0, column=1, pady=10)

csv_label.grid(row=1, column=0, pady=10)
csv_entry.grid(row=1, column=1, pady=10)
csv_button.grid(row=1, column=2, pady=10)

start_button.grid(row=2, column=0, pady=10)
train_button.grid(row=2, column=1, pady=10)

export_txt_button.grid(row=3, column=0, pady=10)
export_csv_button.grid(row=3, column=1, pady=10)

graph_button.grid(row=4, column=0, pady=10)
history_button.grid(row=4, column=1, pady=10)

output_label.grid(row=5, column=0, pady=10)
output_entry.grid(row=5, column=1, pady=10)

# Start the main event loop
root.mainloop()