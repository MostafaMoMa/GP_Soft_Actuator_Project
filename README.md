Data-Driven Modeling and Control of a Pneumatic Soft Actuator Using Gaussian Process Regression

Author: Mostafa Mo. Massoud

Email: mmassoud@stevens.edu || mostafa.mohammed.masssoud@gmail.com

📌 Project Description

Soft robotic actuators exhibit nonlinear, hysteretic behavior that makes traditional control methods difficult.

This project develops a Gaussian Process Regression (GPR) model to:

--> Learn the relationship between PWM input and force output

--> Predict force changes using a data-driven model

--> Evaluate model accuracy (RMSE, MAE, NLL)

--> Demonstrate uncertainty using GP variance

--> Validate hyperparameter optimization and convergence

--> The dataset was collected from a pneumatic soft actuator system using real experimental measurements.

📁 Project Structure
.

├── main.ipynb               # Main notebook integrating all modules

├── gp_model.py              # Gaussian Process + RBF Kernel implementation

├── data_handler.py          # CSV loading, preprocessing, scaling

├── metrics.py               # RMSE, MAE, NLL, error list (comprehension)

├── visualization.py         # Plotting functions

├── test_gp_model.py         # Pytest unit tests for kernel + GP

├── initialDataForGPR100N600Sample.csv   # Dataset

├── README.md                # This file

🧠 Key Features Implemented

✔ Object-Oriented GP model

✔ Custom RBF kernel with noise handling (if logic)

✔ Operator overloading (__mul__)

✔ Magic method (__call__)

✔ List comprehension for error list

✔ Use of enumerate + zip to display predictions

✔ try/except for CSV loading

✔ Pytest for unit testing

🚀 How to Run the Project

1. Open main.ipynb in Jupyter Notebook
   
  Make sure Python 3.12 is installed.
  
2. Install required libraries
   
  Inside your notebook:  pip install numpy pandas matplotlib scipy scikit-learn pytest
  
3. Run the notebook
   
This will:

--> Load and scale the dataset

--> Optimize GP hyperparameters

--> Run convergence loop

--> Evaluate metrics

--> Plot GP predictions and confidence intervals

📊 Example Outputs

--> Hyperparameter optimization results

--> GP predicted vs true force

--> 95% confidence intervals

--> RMSE / MAE / NLL

🧪 Testing

test_gp_model.py includes two tests:

  --> Kernel symmetry check
  
  --> GP training negative log-likelihood check

👤 Contributions

This entire project was completed by Mostafa Mo. Massoud, including:

--> Dataset processing

--> GP model implementation

--> Kernel functions & operator overloading

--> Visualization

--> Pytest unit tests

--> Final notebook integration

--> GitHub repository setup

📬 Contact

If you have questions about the implementation or dataset, contact:

📧 mmassoud@stevens.edu || mostafa.mohammed.massoud@gmail.com

📌 END OF README
