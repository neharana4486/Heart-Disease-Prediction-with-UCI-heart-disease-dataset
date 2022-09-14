# Heart-Disease-Prediction-with-UCI-heart-disease-dataset
In this project UCI heart disease dataset from the UCI Machine Learning data repository was used. Firstly preprocessing of the data was done followed by normalization of the data and feature selection. For SVM classifier model, linear, polynomial and radial basis function (rbf) kernels were used, and parameter tuning was done with ‘gamma’ and ‘C-value’.

To visualize our data models, we used Plotly Dash. It is a rather simple library that allows building and deploying interactive web applications purely in Python. (25) The applications are used through the browser and can operate via local host as well as be deployed and shared with URLs (26). There is a wide repository of Dash documents (open source) with base codes for all needed elements (26) that we edited and used for composing MEDecide clinical decision support tool.
Dash applications are normally comprised of two main segments – layout and callback (27). In our app layout we used dash_core_components and dash_html_components. Some dash_html_components were used for adding titles, text and style.
Figure 3 Best performing SVM for 5 class labels
Figure 2 Best performing MLP scores of 10 fold validation and train test.
As a result of our findings from analyzing the databases we decided to create a MEDecide as a CDS tool that predicts the presence or absence of cardiovascular diseases. This means that binary labels are used instead of severity labelling, due to the previously discussed reasons.
For attributes Maximum heart rate achieved and ST depression induced by exercise relative to rest, we used dash_core_components when creating number input fields. For other attributes Chest pain type, Number of vessels colored by fluoroscopy, the heart status as retrieved from Thallium test andFasting blood sugar, dropdown menus from dash_core_components were selected. The decision to have dropdown menus was based on the fact that those attributes in the dataset had between 2 and 4 options for values, thus having menus from which a user could select suggested values seemed like an optimal choice.
Lead by our curiosity and desire to add some variety, we decided to use a slider for attribute Exercise induced angina which has two possible input options present and not present. The user can only select one value with the slider, which was an important advantage in using it in comparison with checkboxes.
The last component of the layout was a button, which was used to trigger the prediction. The button also allowed us to display a simple explanation regarding the number displayed as a result.
The second part of a Dash application is the allbacks, which are Python functions that are automatically called by Dash when input is entered to the application (28). Callbacks take one or more inputs by their IDs and create one or more outputs according to the called function. This might mean simply writing out a line of text, making calculations or updating a graph.
As our application has a form-like layout where we want the system to start processing input values after the user finished entering all of them, thus avoiding the normal reactiveness of Dash, we decided to create a callback using State. In order to do this dash.dependencies Input, Output and State had to be imported from the dash_html_components.

![Example of a number input field and a dropdown menu](https://github.com/neharana4486Heart-Disease-Prediction-with-UCI-heart-disease-dataset/blob/main/example.JPG?raw=true)

In order to allow the user to trigger the callback when filling in the form has been completed, a submit button was added to the layout. The html.button component has an n_clicks i.e., number of clicks property, which was used to create the second part of the callback, where the output was defined (29).

When the button is clicked, so the number of clicks is above 0, the database is read, and the prediction is carried out. The MLP classifier uses the values entered by the user through the interface of our web application and runs the algorithm in order to predict the presence/absence of cardiovascular disease. MinMaxScaler was used for normalization of the data. Until all the values are entered the user is reminded to enter the missing values and calculation cannot be carried out. Then, the system shows the prediction 0 or 1.
