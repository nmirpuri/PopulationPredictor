
# IMPORTING ALL THE LIBRARIES
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.express as px  
import plotly.graph_objects as go
from numerize import numerize 
st. set_page_config(layout="centered")

st.markdown(
    """
    <style>
    /* Set all text to a smaller size */
    body {
        font-size: 12px;
    }

    /* Adjust the title size */
    .streamlit-expanderHeader {
        font-size: 14px !important;
    }

    /* Adjust headers */
    h1, h2, h3, h4, h5, h6 {
        font-size: 16px !important;
    }

    /* Adjust text in markdown */
    .markdown-text-container {
        font-size: 12px !important;
    }

    /* Adjust input labels */
    .stTextInput label, .stSelectbox label {
        font-size: 12px !important;
    }

    /* Adjust button text */
    .stButton>button {
        font-size: 12px !important;
    }

    </style>
    """, unsafe_allow_html=True)

# IMPORT DATASETS
countries_pop = pd.read_csv(r'datasets/Countries_Population_final.csv')
countries_name= pd.read_csv(r'datasets/Countries_names.csv')

# DASHBOARD TITLE
col1, col2,col3 = st.columns([1,4,1])
with col1:
    pass
with col2:
    st.info('# :blue[POPULATION PREDICTION SYSTEM]')
with col3:
    pass
#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm ML MODEL: POLYNOMIAL REGRESSION mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
col1, col2 = st.columns(2)
with col1:
    # USER INPUTS FOR PREDECTION
    option = st.selectbox(
        'PLEASE SELECT ANY COUNTRY',
        (sorted(countries_name['Country_Name'])))
    year = st.text_input('PLEASE ENTER YEAR', '2030')
    if year.isnumeric():        
        # Divide Independent and Dependent features
        X=countries_pop['Year'] # all the independent features are copied to X
        y=countries_pop[option] # the dependent feature is copied to y

        # Train Test splitting
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        # SVR
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error

        def create_polynomial_regression_model(degree,Yearin):             
            poly_features = PolynomialFeatures(degree=degree)          
            X_train_poly = poly_features.fit_transform(X_train) # transforms the existing features to higher degree features.
                 
            poly_model = LinearRegression() 
            poly_model.fit(X_train_poly, Y_train)    # fit the transformed features to Linear Regression    
            y_train_predicted = poly_model.predict(X_train_poly) # predicting on training data-set
            y_test_predict = poly_model.predict(poly_features.fit_transform(X_test)) # predicting on test data-set
            output = poly_model.predict(poly_features.fit_transform([[Yearin]]))           
        
            # rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted)) # evaluating the model on training dataset
            # r2_train = r2_score(Y_train, y_train_predicted)        
            # rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))  # evaluating the model on test dataset            
            r2_test = r2_score(Y_test, y_test_predict)
        
            # #st.write("RMSE of training set is {}".format(rmse_train))
            # st.write("TRAINING SET ACCURACY {}".format(r2_train))
            # st.write("-------------------------------------------")
            # #st.write("RMSE of test set is {}".format(rmse_test))
            # st.write("TEST SET ACCURACY {}".format(r2_test))
            # st.write("-------------------------------------------")
            st.write("#### :green[ALGORITHM: ] POLYNOMIAL REGRESSION")
            st.write('#### :green[ ACCURACY: ] '+str(int(r2_test*100))+'%')  
            return output

        pred= create_polynomial_regression_model(2,int(year))

        # OUTPUT DETAILS        
        pred_pop = numerize.numerize(pred[0])
        st.write("#### :green[COUNTRY:  ] "+option.upper())
        st.write("#### :green[YEAR:  ] "+year)
        st.write("#### :green[PREDICTED POPULATION:  ] "+pred_pop)
    else:
        st.write('PLEASE ENTER A VALID YEAR')

with col2:
    if year.isnumeric():
        st.write('#### :green['+option.upper()+"'S  POPULATION]")
        fig1 = go.Figure()
        # Create and style traces
        fig1.add_trace(go.Scatter(x=countries_pop['Year'] , y=countries_pop[option], name = "Previous Year's",
                                line=dict(color='green', width=11)
                                ))
        fig1.add_trace(go.Scatter(x=[year], y=[pred[0]], 
                                name = 'Predicted '+year,
                                mode = 'markers',
                                marker_symbol = 'star',
                                marker=dict(
                                size=20, 
                                color=np.random.randn(500), #set color equal to a variable
                                colorscale='reds', # one of plotly colorscales                                
                                )
                                ))
        fig1.update_layout(
            width=400,  # Adjust width of the plot
            height=400  # Adjust height of the plot
        )
        st.plotly_chart(fig1)
        st.write('The above plot shows the population of a country from 1960 to 2021, star represents the predicted population for the given year')
