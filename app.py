import streamlit as st

# train a model for iris classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# make predictions and show the iris class
def iris_classification(features):
    pred = clf.predict([features])
    return iris.target_names[pred][0]





def main():

    # create a side bar
    st.sidebar.title("Introduction to Streamlit App")

    # take radio input
    page = st.sidebar.radio("Select a page", ["Introduction", "Input & Output Methods", "Iris Classification"])

    if page == "Introduction":
        st.write("Hello, world!")
        st.write("This is a Streamlit app created by me, Bhawani Shankar.")
    
    elif page == "Input & Output Methods":
        st.title("Input & Output Methods")
        # take radio input
        user_type = st.radio("Select a user type", ["Student", "Teacher", "Parent"])
        # take text input
        name = st.text_input("Enter your name")
        # take number input
        age = st.number_input("Enter your age", min_value=0, max_value=100)
        # take date input
        dob = st.date_input("Enter your date of birth")

        # take input using a slider
        weight = st.slider("Enter your weight", min_value=2, max_value=200)
        # take height of the user in slider
        height = st.slider("Enter your height", min_value=2, max_value=200)

        # show the bmi of the user
        bmi = weight / ((height/100) ** 2)
        st.write(f"Your BMI is {bmi}")

    elif page == "Iris Classification":
        st.title("Iris Classification App")
        # take input from the user
        sepal_length = st.number_input("Enter the sepal length")
        sepal_width = st.number_input("Enter the sepal width")
        petal_length = st.number_input("Enter the petal length")
        petal_width = st.number_input("Enter the petal width")

        # show the iris class
        if st.button("Predict Class"):
            features = [sepal_length, sepal_width, petal_length, petal_width]
            iris_class = iris_classification(features)
            st.write(f"The iris class is {iris_class}")


    

if __name__ == "__main__":
    main()

