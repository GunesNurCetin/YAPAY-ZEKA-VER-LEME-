#GÜNEŞ NUR ÇETİN


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import boxplot_stats
import random
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# VERİ OKUMA
nitelikAdlari = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]

veriSeti = pd.read_csv("auto-mpg.data", sep="\s+", header=None, names=nitelikAdlari, decimal=".")

# Eksik veri kontrolü
print(veriSeti.isnull().sum())

# EKSIK VERİNİN TAMAMLANMASI (Missing Data Imputation)
# 'horsepower' sütunundaki '?' karakterlerini NaN ile değiştir
veriSeti.loc[veriSeti.horsepower == "?", "horsepower"] = np.nan
veriSeti.horsepower = veriSeti.horsepower.astype("float64")

# Sayısal bir nitelikteki eksik verinin tamamlanması
veriSeti["horsepower"].fillna(veriSeti["horsepower"].mean().round(0), inplace=True)

# Kategorik bir nitelikteki eksik verinin tamamlanması
# (Bu örnekte 'renk' sütunu yok, dolayısıyla bu kısmı kaldırıyoruz)
# veriSeti["renk"].fillna(veriSeti["renk"].mode()[0], inplace=True)

# VERİ AYRIŞTIRMA
# I. yol:
veriSeti["durum"] = veriSeti.mpg.map(lambda x: "Düşük" if x < 23.5 else "Orta" if ((x >= 23.5) & (x < 30)) else "Yüksek").astype("category")
print(veriSeti.durum.value_counts())

# II. yol:
# I. Yöntem: Sabit aralıklar
bolmeKategorileri = ["Düşük", "Orta", "Yüksek"]
bolmeler = [8, 23.4, 29.9, 46.6]
veriSeti["durum"] = pd.cut(veriSeti["mpg"], bins=bolmeler, labels=bolmeKategorileri)
print(veriSeti.durum.value_counts())

# II. Yöntem: Eşit frekans
durum_ef = pd.qcut(veriSeti["mpg"], q=3)
print(durum_ef.value_counts())

# III. Yöntem: Eşit aralıklar
durum_ea = pd.cut(veriSeti["mpg"], bins=3)
print(durum_ea.value_counts())

# VERİ ÖZETLEME
pd.set_option("display.max_columns", 20)
print(veriSeti.describe(include="all"))

# GRUPLANDIRMA (Aggregate)
print(veriSeti[["mpg", "durum"]].groupby("durum").mean())
print(veriSeti[["mpg", "durum"]].groupby("durum").sum())
print(veriSeti[["mpg", "durum"]].groupby("durum").count())
print(veriSeti[["mpg", "durum"]].groupby("durum").min())
print(veriSeti[["mpg", "durum"]].groupby("durum").max())
print(veriSeti[["mpg", "durum"]].groupby("durum").std())
print(veriSeti[["mpg", "durum"]].groupby("durum").describe())

# TEKRAR EDEN SATIRLARIN BULUNMASI
tekrarlar_f = veriSeti.duplicated(keep="first")
tekrarlar_l = veriSeti.duplicated(keep="last")

# Tekrar eden tüm gözlemleri görmek için
indislerim = tekrarlar_f | tekrarlar_l
print(veriSeti[indislerim])

# Veri setinden tekrar eden değerleri kaldırma
veriSeti2 = veriSeti.drop_duplicates()

# Aykırı Değerlerin Tespiti ve Temizlenmesi
q1 = veriSeti["horsepower"].quantile(0.25)
q3 = veriSeti["horsepower"].quantile(0.75)
IQR = q3 - q1
alt = q1 - 1.5 * IQR
ust = q3 + 1.5 * IQR

ust_aykiriDegerInd = np.where(veriSeti["horsepower"] >= ust)[0]
alt_aykiriDegerInd = np.where(veriSeti["horsepower"] <= alt)[0]

veriSeti.drop(index=ust_aykiriDegerInd, inplace=True)
veriSeti.drop(index=alt_aykiriDegerInd, inplace=True)

sns.boxplot(y="horsepower", data=veriSeti, palette="magma")

# II. yol: Boxplot ile aykırı değerlerin temizlenmesi
aykiriDegerler = boxplot_stats(veriSeti.horsepower).pop(0)["fliers"]
aykiriDegerIndeksleri = veriSeti.index[veriSeti.horsepower.isin(aykiriDegerler) == True]
veriSeti = veriSeti.drop(aykiriDegerIndeksleri, axis=0)

# ÖRNEKLEME (Sampling)
listem = list(np.arange(1, 21))

print(sample(listem, 10)) # Kod 1. kez çalıştırılıyor
print(sample(listem, 10)) # Kod 2. kez çalıştırılıyor

random.seed(123)
print(sample(listem, 10)) # Kod 3. kez çalıştırılıyor
random.seed(123)
print(sample(listem, 10)) # Kod 4. kez çalıştırılıyor

random.seed(5)
print(sample(listem, 10))
random.seed(5)
print(sample(listem, 10))

# Yerine koyarak seçim
print(random.choices(listem, k=10))

# Basitçe eğitim ve test veri setleri yaratmak
egitim = veriSeti.sample(frac=0.7, replace=False, random_state=1)
ind = veriSeti.index.isin(egitim.index)
test = veriSeti[~ind]

# Tabakalı Örnekleme (Stratified Sampling)
egitim1, test1 = train_test_split(veriSeti, train_size=0.7, stratify=veriSeti["durum"], random_state=1)

print(veriSeti.durum.value_counts())
print(egitim1.durum.value_counts())
print(test1.durum.value_counts())

# YAPAY KODLAMA (Dummy Coding)
veriSeti["durum_s1"] = veriSeti["durum"].cat.codes
print(veriSeti.durum.value_counts())
print(veriSeti.durum_s1.value_counts())

durum_s2 = pd.get_dummies(veriSeti.durum, columns=["durum"], dtype=int)
veriSeti = pd.concat([veriSeti, durum_s2], axis=1)

# VERİ NORMALİZASYONU (Data Normalization)
# Min-max normalizasyon yöntemi
veri = veriSeti.iloc[:, 0:8]
scaler = MinMaxScaler()
scaler.fit(veri)
n_veriSeti = scaler.transform(veri)
n_veriSeti = pd.DataFrame(n_veriSeti, columns=veri.columns)

# Standartizasyon (Z-score)
stScaler = StandardScaler()
s_veriSeti = stScaler.fit_transform(veri)
s_veriSeti = pd.DataFrame(s_veriSeti, columns=veri.columns)
