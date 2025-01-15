import streamlit as st
import pandas as pd
import joblib  # Untuk menyimpan model dalam format .sav
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Contoh model

# Menambahkan judul aplikasi
st.title("Student Performance Prediction")

# Opsi untuk mengunggah file
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Membaca file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)

        st.write("Preview of the data:")
        st.dataframe(data.head())

        # Pemrosesan awal (contoh)
        if st.button("Train Model"):
            with st.spinner("Processing..."):
                # Split data
                st.write("Splitting data...")
                X = data.iloc[:, :-1]  # Semua kolom kecuali yang terakhir sebagai fitur
                y = data.iloc[:, -1]   # Kolom terakhir sebagai target
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Melatih model
                st.write("Training model...")
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Evaluasi model
                accuracy = model.score(X_test, y_test)
                st.write(f"Model accuracy: {accuracy:.2f}")

                # Menyimpan model sebagai .sav
                model_filename = "student_performance_model.sav"
                joblib.dump(model, model_filename)
                st.success(f"Model saved as {model_filename}")

                # Visualisasi contoh
                st.write("Feature Importances:")
                feature_importances = model.feature_importances_
                fig, ax = plt.subplots()
                ax.barh(X.columns, feature_importances)
                ax.set_title("Feature Importances")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Informasi tambahan
st.info("This application trains a RandomForest model and saves it as .sav.")
