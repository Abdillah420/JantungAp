import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from io import StringIO
import gdown

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'kfold_data' not in st.session_state:
    st.session_state['kfold_data'] = None

# Sidebar untuk navigasi
st.sidebar.title("WEB PREDIKSI PENYAKIT JANTUNG ")
st.sidebar.markdown("---")
st.sidebar.markdown("---")

# Pilihan halaman
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Unggah dan Eksplorasi Data", "Visualisasi Data", "Preprocessing Data", "Model Klasifikasi", "Prediksi"]
)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi untuk Analisis dan Prediksi Risiko Penyakit Jantung.")

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'])
    plt.title(title)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(plt)

# Halaman Unggah dan Eksplorasi Data
if page == "Unggah dan Eksplorasi Data":
    st.title("Unggah dan Eksplorasi Data")
    
    # Tambahkan opsi untuk memilih sumber data
    data_source = st.radio(
        "Pilih Sumber Data",
        ["Upload File dari Local", "Google Drive developer (tinggal pencet tanpa pilih pilih file)"]
    )
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
    
    else:  # Google Drive
        # URL Google Drive
        url = "https://drive.google.com/file/d/1J9r1Iup_gxxQTtNfgMNaiRBV638F3n7Y/view?usp=sharing"
        
        if st.button("Load Data dari Google Drive Abdillah"):
            try:
                # Ubah URL menjadi format yang bisa didownload
                file_id = url.split('/')[-2]
                download_url = f'https://drive.google.com/uc?id={file_id}'
                
                # Download file
                output = 'heart.csv'
                gdown.download(download_url, output, quiet=False)
                
                # Baca file CSV
                data = pd.read_csv(output)
                st.session_state['data'] = data
                st.success("Data berhasil dimuat dari Google Drive!")
            
            except Exception as e:
                st.error(f"Error saat memuat data: {str(e)}")
    
    # Tampilkan informasi data jika sudah dimuat
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        
        st.subheader("Data yang Dimuat")
        st.write(data)
        
        st.subheader("Informasi Data")
        buffer = StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        st.subheader("Deskripsi Statistik")
        st.write(data.describe())
        
        st.subheader("Pemeriksaan Nilai yang Hilang")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0])

# Halaman Visualisasi Data
elif page == "Visualisasi Data":
    st.title("Visualisasi Data")
    
    if st.session_state['data'] is None:
        st.warning("Unggah data terlebih dahulu di halaman 'Unggah dan Eksplorasi Data'.")
    else:
        data = st.session_state['data']
        
        st.subheader("Histogram Fitur")
        feature = st.selectbox("Pilih Fitur untuk Histogram", data.columns)
        plt.figure(figsize=(10, 5))
        sns.histplot(data[feature], kde=True)
        st.pyplot(plt)
        
        st.subheader("Korelasi Antar Fitur")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

# Halaman Preprocessing Data
elif page == "Preprocessing Data":
    st.title("Preprocessing Data")
    
    if st.session_state['data'] is None:
        st.warning("Unggah data terlebih dahulu di halaman 'Unggah dan Eksplorasi Data'.")
    else:
        data = st.session_state['data']
        
        st.subheader("Sebelum Preprocessing")
        st.write(data.head())
        
        # Penanganan nilai yang hilang
        st.subheader("Penanganan Nilai yang Hilang")
        if st.checkbox("Isi nilai yang hilang dengan rata-rata"):
            data = data.fillna(data.mean())
            st.success("Nilai yang hilang telah diisi")
        
        # Encoding variabel kategorikal
        st.subheader("Encoding Variabel Kategorikal")
        categorical_columns = data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            data = pd.get_dummies(data, columns=categorical_columns)
            st.success("Encoding selesai")
        
        # Pemisahan fitur dan target
        X = data.drop('output', axis=1)
        y = data['output']
        
        # K-Fold Cross Validation
        kf = KFold(n_splits=8, shuffle=True, random_state=42)
        
        # Simpan data untuk penggunaan di halaman lain
        st.session_state['kfold_data'] = {
            'X': X,
            'y': y,
            'kf': kf
        }
        
        st.success("Data telah siap untuk pemodelan dengan 8-Fold Cross Validation")

# Halaman Model Klasifikasi
elif page == "Model Klasifikasi":
    st.title("Model Klasifikasi")
    
    if st.session_state['kfold_data'] is None:
        st.warning("Lakukan preprocessing data terlebih dahulu")
    else:
        X = st.session_state['kfold_data']['X']
        y = st.session_state['kfold_data']['y']
        kf = st.session_state['kfold_data']['kf']
        
        models = {
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        
        for name, model in models.items():
            st.subheader(f"Model: {name}")
            
            fold_accuracies = []
            all_cms = np.zeros((2, 2))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Konversi indeks ke numpy array
                train_idx = np.array(train_idx)
                val_idx = np.array(val_idx)
                
                # Ambil data untuk fold saat ini
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Standardisasi
                scaler = StandardScaler()
                X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                X_val_fold_scaled = scaler.transform(X_val_fold)
                
                try:
                    # Training dan evaluasi
                    model.fit(X_train_fold_scaled, y_train_fold)
                    y_pred = model.predict(X_val_fold_scaled)
                    
                    accuracy = accuracy_score(y_val_fold, y_pred)
                    cm = confusion_matrix(y_val_fold, y_pred)
                    
                    fold_accuracies.append(accuracy)
                    all_cms += cm
                    
                    st.write(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
                except Exception as e:
                    st.error(f"Error pada fold {fold+1}: {str(e)}")
                    continue
            
            if fold_accuracies:
                mean_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                st.write(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
                
                st.write("Aggregate Confusion Matrix:")
                
                # Pastikan confusion matrix dalam format integer
                all_cms = all_cms.astype(int)
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(all_cms, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Negatif', 'Positif'],
                           yticklabels=['Negatif', 'Positif'])
                plt.title(f"Aggregate Confusion Matrix untuk {name}")
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(plt)
                plt.close()

# Halaman Prediksi
elif page == "Prediksi":
    st.title("Prediksi Risiko Penyakit Jantung")
    
    if st.session_state['kfold_data'] is None:
        st.warning("Lakukan preprocessing dan pelatihan model terlebih dahulu")
    else:
        st.subheader("Masukkan Data untuk Prediksi")
        
        # Input fields
        age = st.number_input("Usia (age)", min_value=0)
        sex = st.number_input("Jenis Kelamin (sex, 0=Perempuan, 1=Laki-laki)", min_value=0, max_value=1)
        cp = st.number_input("Tipe Nyeri Dada (cp, 0-3)", min_value=0, max_value=3)
        trtbps = st.number_input("Tekanan Darah (trtbps)", min_value=0)
        chol = st.number_input("Kadar Kolesterol (chol)", min_value=0)
        fbs = st.number_input("Gula Darah Puasa (fbs, 0=Tidak, 1=Ya)", min_value=0, max_value=1)
        restecg = st.number_input("Hasil Elektrokardiografi (restecg, 0-2)", min_value=0, max_value=2)
        thalachh = st.number_input("Detak Jantung Maksimum (thalachh)", min_value=0)
        exng = st.number_input("Nyeri Dada Saat Berolahraga (exng, 0=Tidak, 1=Ya)", min_value=0, max_value=1)
        oldpeak = st.number_input("Oldpeak", min_value=0.0)
        slp = st.number_input("Kemampuan Puncak (slp, 0-2)", min_value=0, max_value=2)
        caa = st.number_input("Jumlah Pembuluh Darah (caa, 0-3)", min_value=0, max_value=3)
        thal = st.number_input("Kondisi Thalassemia (thal, 1-3)", min_value=1, max_value=3)
        
        if st.button("Prediksi"):
            # Prepare input data
            input_data = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, 
                          exng, oldpeak, slp, caa, thal]]
            
            # Scale input data
            scaler = StandardScaler()
            X = st.session_state['kfold_data']['X']
            scaler.fit(X)
            input_data_scaled = scaler.transform(input_data)
            
            # Make prediction using KNN
            knn = KNeighborsClassifier()
            knn.fit(scaler.transform(X), st.session_state['kfold_data']['y'])
            prediction = knn.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.warning("Prediksi: Positif Risiko Penyakit Jantung")
            else:
                st.success("Prediksi: Negatif Risiko Penyakit Jantung")