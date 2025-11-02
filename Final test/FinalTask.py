# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set(style="whitegrid")
print('Current working dir:', os.getcwd())
print('Files in workspace (top 40):')
print(os.listdir(os.getcwd())[:40])

candidates = ['loan_data_2007_2014_cleaned.csv', 'loan_data_2007_2014.csv']
data_file = None
for f in candidates:
    if os.path.exists(f):
        data_file = f
        break

if data_file is None:
    raise FileNotFoundError('Tidak menemukan dataset di workspace root. Letakkan CSV di folder proyek atau ubah path.')

print('Loading', data_file)
df = pd.read_csv(data_file, low_memory=False)
print('Loaded shape:', df.shape)

display(df.head())

sample = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df.copy()
print('Sample shape for exploratory work:', sample.shape)

# %%

print('Columns:', len(df.columns))
print('\nColumn dtypes summary:')
display(df.dtypes.value_counts())

# unique counts for top columns (helpful to spot ids vs categories)
unique_counts = df.nunique(dropna=False).sort_values(ascending=False)
display(unique_counts.head(50))

print('\nNumeric description:')
display(df.select_dtypes(include=[np.number]).describe().T)

print('\nNon-numeric sample description:')
display(df.select_dtypes(exclude=[np.number]).head())

# %%
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
missing = pd.concat([missing_count, missing_pct], axis=1)
missing.columns = ['missing_count', 'missing_pct']
missing = missing[missing['missing_count'] > 0].sort_values('missing_count', ascending=False)
print('Columns with missing values:')
display(missing.head(100))

# Visualize top missing columns if present
if not missing.empty:
    plt.figure(figsize=(8, min(6, len(missing))))
    sns.barplot(x='missing_pct', y=missing.index[:40], data=missing.reset_index().rename(columns={'index':'col'}).iloc[:40])
    plt.xlabel('% missing')
    plt.title('Top missing columns')
    plt.show()

# %%
numeric = df.select_dtypes(include=[np.number])
print('Numeric columns count:', numeric.shape[1])

if numeric.shape[1] > 0:
    desc = numeric.describe().T
    desc['iqr'] = desc['75%'] - desc['25%']
    desc['outlier_rule_upper'] = desc['75%'] + 1.5 * desc['iqr']
    desc['outlier_rule_lower'] = desc['25%'] - 1.5 * desc['iqr']
    display(desc[['count','mean','std','min','25%','50%','75%','max','iqr','outlier_rule_lower','outlier_rule_upper']].head(50))

    # plot up to 12 histograms
    cols = list(numeric.columns)[:12]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    for ax, c in zip(axes, cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, bins=50)
        ax.set_title(c)
    for ax in axes[len(cols):]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()
else:
    print('No numeric columns found.')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df['loan_amnt'], bins=40, kde=True, color='skyblue')
plt.title('Distribusi Jumlah Pinjaman')
plt.xlabel('Jumlah Pinjaman (loan_amnt)')
plt.ylabel('Frekuensi')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='loan_status', y='int_rate', data=df, palette='Set2')
plt.title('Hubungan Tingkat Bunga dengan Status Kredit')
plt.xlabel('Status Kredit')
plt.ylabel('Tingkat Bunga (%)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='loan_status', y='dti', data=df, palette='Set3')
plt.title('Hubungan Debt-to-Income (DTI) dengan Status Kredit')
plt.xlabel('Status Kredit')
plt.ylabel('Debt-to-Income Ratio')
plt.xticks(rotation=45)
plt.show()




# %%

    numeric_sample = sample.select_dtypes(include=[np.number]) if 'sample' in globals() else df.select_dtypes(include=[np.number])
    n = numeric_sample.shape[1]
    print('Numeric columns considered for correlation:', n)
    if n <= 1:
        print('Not enough numeric columns for correlation analysis.')
    else:
        corr = numeric_sample.corr()
        import numpy as _np
        # user-tunable: how many top pairs to show
        top_k = 15
        # compute upper triangle (exclude diagonal) and stack
        mask_u = _np.triu(_np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask_u)
        stacked = upper.stack()  # pairs with i<j
        if stacked.empty:
            print('No variable pairs found (maybe all NaN correlations).')
        else:
            # sort by absolute value and take top_k
            s_abs = stacked.abs().sort_values(ascending=False)
            k = min(top_k, len(s_abs))
            top = s_abs.iloc[:k].reset_index()
            top.columns = ['var1','var2','abs_corr']
            # retrieve signed correlation
            top['corr'] = top.apply(lambda r: corr.at[r['var1'], r['var2']], axis=1)
            # display concise table
            display(top[['var1','var2','corr','abs_corr']])
            # barplot of absolute correlations
            plt.figure(figsize=(8, max(2, k * 0.4)))
            labels = top.apply(lambda r: f"{r['var1']} ↔ {r['var2']}" , axis=1)
            sns.barplot(x='abs_corr', y=labels, data=top, palette='viridis')
            plt.xlabel('Absolute correlation')
            plt.title(f'Top {k} variable pairs by |correlation|')
            plt.tight_layout()
            plt.show()
            # small heatmap for involved variables if subset is small
            involved = pd.unique(top[['var1','var2']].values.ravel())
            if len(involved) <= 20:
                sub = corr.loc[involved, involved]
                plt.figure(figsize=(max(4, len(involved)*0.6), max(3, len(involved)*0.6)))
                sns.heatmap(sub, annot=True, fmt='.2f', cmap='vlag', center=0, linewidths=0.2, cbar_kws={'shrink':0.7})
                plt.title('Correlation matrix for variables in top pairs')
                plt.tight_layout()
                plt.show()
            else:
                print(f'Skipping small heatmap: {len(involved)} unique variables in top pairs (too many to display).')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Deteksi kolom target ---
possible_targets = ['loan_status']
found = [c for c in possible_targets if c in df.columns]

if found:
    target = found[0]
    print(f"Kolom target terdeteksi: {target}")
    print("\nDistribusi nilai target:")
    print(df[target].value_counts(dropna=False))
    
    # --- 2. Buat grafik distribusi ---
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target, palette='pastel', edgecolor='black')
    plt.title(f"Distribusi Nilai Kolom Target: {target}", fontsize=12)
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Kolom target tidak ditemukan.")


# %% [markdown]
# # Data Preparation

# %%
import pandas as pd
import numpy as np

# --- 1. Hapus kolom duplikat ---
df = df.loc[:, ~df.columns.duplicated()]

# --- 2. Hapus kolom dengan missing value > 40% ---
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.4].index
print(f"🧹 Menghapus {len(cols_to_drop)} kolom karena missing value > 40%")
df.drop(columns=cols_to_drop, inplace=True)

# --- 3. Hapus baris duplikat ---
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"🧹 Menghapus {before - df.shape[0]} baris duplikat")

# --- 4. Normalisasi nama kolom ---
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('[^a-z0-9_]', '', regex=True)
)

# --- 5. Bersihkan kolom numerik yang tersimpan sebagai string ---
for col in df.columns:
    if df[col].dtype == 'object':
        # Contoh: ubah "36 months" → 36
        if df[col].str.contains('month', case=False, na=False).any():
            df[col] = df[col].str.extract(r'(\d+)').astype(float)
        # Contoh: ubah "10%" → 0.10
        elif df[col].str.contains('%', na=False).any():
            df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100

# --- 6. Tangani nilai kosong ---
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# Untuk kolom numerik → isi dengan median
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# Untuk kolom kategorikal → isi dengan modus
for col in cat_cols:
    if df[col].isnull().any():
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

# --- 7. Hapus kolom yang bersifat ID atau tidak relevan ---
id_cols = ['id', 'member_id', 'url', 'title', 'policy_code']
df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True, errors='ignore')

# --- 8. Tampilkan ringkasan hasil cleaning ---
print("\n📊 Info Dataset Setelah Cleaning:")
df.info()

print("\n🧩 Missing value tersisa per kolom:")
missing = df.isnull().sum()
print(missing[missing > 0])

# --- 9. Simpan hasil cleaning ---
output_path = "loan_data_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Data cleaning selesai. File disimpan sebagai: {output_path}")


# %%
# === 1. Labelisasi ===
if 'loan_status' in df.columns:
    good_status = [
        'fully paid',
        'current',
        'does not meet the credit policy. status:fully paid',
        'in grace period'
    ]
    
    df['loan_status_clean'] = df['loan_status'].astype(str).str.strip().str.lower()
    df['credit_label'] = df['loan_status_clean'].apply(lambda x: 0 if x in good_status else 1)
    
    print("\n✅ Labelisasi selesai.")
    print(df['credit_label'].value_counts())
else:
    print("❌ Kolom 'loan_status' tidak ditemukan dalam dataset.")

# === 2. Visualisasi ===
import matplotlib.pyplot as plt

if 'credit_label' in df.columns:
    label_counts = df['credit_label'].value_counts().sort_index()
    labels = ['Good Loan (0)', 'Bad Loan (1)']
    colors = ['#4CAF50', '#F44336']

    # Bar chart
    plt.figure(figsize=(6,4))
    plt.bar(labels, label_counts, color=colors)
    plt.title('Distribusi Good vs Bad Loan')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    # Pie chart
    plt.figure(figsize=(5,5))
    plt.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Proporsi Good vs Bad Loan')
    plt.show()
else:
    print("❌ Kolom 'credit_label' belum tersedia. Jalankan labelisasi terlebih dahulu.")


# %% [markdown]
# ## Endocding

# %%
from sklearn.preprocessing import LabelEncoder

# --- 1. Deteksi kolom kategorikal ---
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Kolom kategorikal yang akan diencoding ({len(cat_cols)}): {cat_cols}")

# --- 2. Label Encoding untuk kolom ordinal ---
ordinal_cols = ['grade', 'sub_grade']
for col in ordinal_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# --- 3. Tentukan kolom dengan banyak kategori (gunakan Frequency Encoding) ---
high_cardinality_cols = [col for col in cat_cols if df[col].nunique() > 20 and col not in ordinal_cols]

for col in high_cardinality_cols:
    freq_map = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq_map)
    print(f"Frequency encoding diterapkan pada: {col} ({df[col].nunique()} nilai unik)")

# --- 4. Kolom kategorikal dengan sedikit kategori (gunakan One-Hot Encoding) ---
low_cardinality_cols = [col for col in cat_cols if col not in ordinal_cols + high_cardinality_cols]
df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)

# --- 5. Informasi hasil ---
print("\n✅ Encoding selesai.")
print(f"Jumlah kolom setelah encoding: {df.shape[1]}")

# --- 6. Cek hasil ---
print("\nBeberapa kolom hasil encoding:")
print(df.head(3))


# %% [markdown]
# ## SMOTE

# %%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Pisahkan fitur (X) dan target (y) ---
X = df.drop(columns=['credit_label'])
y = df['credit_label']

# --- 2. Split data sebelum SMOTE (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Distribusi awal sebelum SMOTE:")
print(y_train.value_counts())

# --- 3. Terapkan SMOTE hanya pada data training ---
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nDistribusi setelah SMOTE:")
print(y_train_res.value_counts())

# --- 4. Tampilkan perbandingan grafik ---
plt.figure(figsize=(6,4))
plt.bar(['Good Loan (0)', 'Bad Loan (1)'], 
        y_train.value_counts().sort_index(), 
        color='#9E9E9E', label='Sebelum SMOTE')
plt.bar(['Good Loan (0)', 'Bad Loan (1)'], 
        y_train_res.value_counts().sort_index(), 
        alpha=0.6, color=['#4CAF50', '#F44336'], 
        label='Setelah SMOTE')
plt.title('Perbandingan Jumlah Data Sebelum & Setelah SMOTE (Split 70/30)')
plt.ylabel('Jumlah Sampel')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %% [markdown]
# # Modeling

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Inisialisasi model ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# --- 2. Latih model pada data hasil SMOTE ---
log_reg.fit(X_train_res, y_train_res)

# --- 3. Prediksi pada data testing ---
y_pred = log_reg.predict(X_test)

# --- 4. Evaluasi hasil ---
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}\n")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)


# %%
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5,1.5], ['Good Loan (0)', 'Bad Loan (1)'])
plt.yticks([0.5,1.5], ['Good Loan (0)', 'Bad Loan (1)'])
plt.tight_layout()
plt.show()



plt.figure(figsize=(5,3))
sns.countplot(x=y_pred, palette=['#4CAF50','#F44336'])
plt.title('Distribusi Prediksi Model')
plt.xlabel('Kategori Prediksi')
plt.ylabel('Jumlah')
plt.xticks([0,1], ['Good Loan (0)', 'Bad Loan (1)'])
plt.tight_layout()
plt.show()


