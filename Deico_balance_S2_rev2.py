"""
Raw datadan sinüs dalgasının başlangıç ve bitiş zamanları alınır, Kare dalga sinyali alınır ve
her iki dalga üst üste çakıştırılır, sonrasında her iki kare dalga arasındaki faz farkları hesaplanarak , balance hesabı yapılır.
Sİnüs dalgasının amplitude değeri , Bu dalgaya karşılık gelen Spectrum datasından okunur, dalganın genliği ise takometreden
elde edilen rpm verisine göre ortalama bir değere göre belirlenir.

Amplitude                -------> Spectrum.xlsx
Period                   -------> rpm from pulse.txt
Inıtial and Final Time   -------> Raw.xlsx

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime
import json
import cmath
matplotlib.use("TkAgg")

# JSON file to store computed results
DATA_FILE = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\data.json"
DATA_FILE_2 = "C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\data2.json"

# Faz açılarını depolamak için liste oluştur
phase_angle_list = []
rpm_measured_list = []
f_measured_list = []
# İşlenecek dosya çiftleri
file_pairs = [
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw1.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse01.txt"),
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw2.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse02.txt"),
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw11.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse11.txt"),
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw12.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse12.txt"),
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw21.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse21.txt"),
    (r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Raw22.xlsx", r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\pulse22.txt"),
]

# Zaman formatını dönüştürme fonksiyonu
def convert_time_format(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")  # 24 saat formatı
    return time_obj.strftime("%H:%M:%S:%f")[:-3]  # Mili saniyeye çevir

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print("Data file not found or invalid JSON!")
        return {}

# Load data from the json file
data = load_data()


# Her dosya çifti için işlemleri yap
for i, (excel_file_path, pulse_file_path) in enumerate(file_pairs, start=1):
    # Excel dosyasını oku
    xls = pd.ExcelFile(excel_file_path)
    sheet_name = "Sensor 1" if "Sensor 1" in xls.sheet_names else "Sensor 3"
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)

    # Zaman sütununu al (3. sütun, index=2)
    time_column = df.iloc[3:, 2].dropna().values
    initial_time_str = convert_time_format(str(time_column[0]))
    end_time_str = convert_time_format(str(time_column[-1]))

    # Başlangıç ve bitiş zamanlarını saniyeye çevirme
    hh, mm, ss, ms = map(int, initial_time_str.split(":"))
    initial_time_seconds = hh * 3600 + mm * 60 + ss + ms / 1000
    hh_end, mm_end, ss_end, ms_end = map(int, end_time_str.split(":"))
    end_time_seconds = hh_end * 3600 + mm_end * 60 + ss_end + ms_end / 1000

    # Pulse dosyasını okuma
    with open(pulse_file_path, "r") as file:
        lines = file.readlines()
    start_index = next(i for i, line in enumerate(lines) if "Sending file contents to the computer..." in line) + 1
    data_lines = lines[start_index:]

    pulse_times = []
    pulse_values = []

    for line in data_lines:
        try:
            parts = line.strip().split(", ")
            time_part = parts[0].split(" ")[1]
            offset_part = parts[1].replace("+", "").replace("ms", "")
            pulse_value = int(parts[2])

            hh, mm, ss_ms = time_part.split(":")
            ss, ms = ss_ms.split(".")
            total_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000 + int(offset_part) / 1000

            pulse_times.append(total_seconds)
            pulse_values.append(pulse_value)
        except Exception as e:
            print(f"Skipping line due to error: {e}")

    # Faz açısı hesaplama
    rising_edge_times = [pulse_times[i] for i in range(1, len(pulse_values)) if
                         pulse_values[i] == 1 and pulse_values[i - 1] == 0]
    periods = np.diff(rising_edge_times)
    T_measured = np.mean(periods)

    if T_measured > 0:
        f_measured = 1 / T_measured
        f_measured_list.append(f_measured)
        rpm_measured = f_measured * 60
        rpm_measured_list.append(rpm_measured)
    else:
        rpm_measured = 0

    # Sinüs dalgası üretme
    f_rpm = rpm_measured / 60
    T_rpm = 1 / f_rpm
    w = 2 * np.pi * f_rpm

    total_duration = end_time_seconds - initial_time_seconds
    t = np.linspace(0, total_duration, 100000)
    A = 1
    y = A * np.sin(w * t)
    t_real = initial_time_seconds + t

    A_time = next((pulse_times[i] for i in range(len(pulse_values)) if pulse_values[i] == 1), None)
    B_time = None
    for j in range(1, len(t_real) - 1):
        if t_real[j] < A_time and y[j - 1] < y[j] > y[j + 1]:
            B_time = t_real[j]

    if A_time is not None and B_time is not None:
        dT = A_time - B_time
        phase_angle = (dT / T_measured) * 360
        phase_angle_list.append(phase_angle)
    else:
        phase_angle_list.append(None)
"""
    # Her veri seti için ayrı figür oluşturma
    plt.figure(figsize=(10, 5))
    plt.plot(t_real, y, label="Sinüs Dalgası", color="blue")
    for j in range(len(pulse_times) - 1):
        plt.plot([pulse_times[j], pulse_times[j + 1]], [pulse_values[j], pulse_values[j]], color="red", linewidth=1)
        plt.plot([pulse_times[j + 1], pulse_times[j + 1]], [pulse_values[j], pulse_values[j + 1]], color="red",
                 linewidth=1)
    plt.axvline(A_time, color='purple', linestyle='--', label="A Noktası (Pulse İlk 1)")
    plt.axvline(B_time, color='orange', linestyle='--', label="B Noktası (Sinüs İlk Peak)")
    plt.xlabel("Gerçek Zaman (s)")
    plt.ylabel("Genlik")
    plt.title(f"Figure {i}: Faz Açısı Gösterimi\nFaz Açısı: {phase_angle:.2f}°")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid()
    plt.show()
"""



Hz_shaft=f_measured_list
dt_phase_list = phase_angle_list
print("Phase Angles:", phase_angle_list)
print("f_measured_list:",f_measured_list)
print("Hz_shaft:",Hz_shaft)
print("rpm_measured:",rpm_measured_list)







print("---------------------------------   Spectrum Data -------------------------------------------------------------")
def process_excel_data(file_name, sheet_name, Hz_shaft):
    # Excel dosyasını oku
    df = pd.read_excel(file_name, sheet_name=sheet_name)

    # 8. satırdaki verileri al (Python indekslemeye göre satır 7)
    row_8_data = df.iloc[6]

    # İstenilen aralıktaki değerleri taramak için başlangıç ve bitiş değerlerini tanımla
    start_range = Hz_shaft - 3
    end_range = Hz_shaft + 3

    # Aralıkta olan değerleri işlemek için filtrele
    averages = []
    for column in df.columns[2:]:  # İlk iki sütunu atla (3. sütundan itibaren başla)
        # 8. satırdaki hücre
        main_value = df[column].iloc[6]  # 8. satırdaki değer

        # Eğer değer istenilen aralıktaysa işleme devam et
        if start_range <= main_value <= end_range:
            # Bu hücrenin altındaki 5 veriyi al
            below_values = df[column].iloc[7:30]  # 8. satırdan 30. satıra kadar
            average = below_values.mean()  # Ortalama hesapla

            # Ortalamayı ve ana hücreyi bir listeye ekle
            averages.append((main_value, average))

    # Eğer seçilen aralıkta veri yoksa uyarı ver
    if not averages:
        print("Belirtilen aralıkta işlem yapılacak değer bulunamadı.")
        return None, None
    else:
        # En yüksek ortalamayı bul
        max_average = max(averages, key=lambda x: x[1])
        #print(f"Frequency: {max_average[0]:.8f}, Amplitude: {max_average[1]:.8f}")
        return max_average[0], max_average[1]

data_amplitude=[]
data_frequency=[]


print(Hz_shaft)
print("---------------------------------------------")
file_name_1 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum1.xlsx" # x_1
sheet_name_1 = "Sensor 1"
frequency, amplitude = process_excel_data(file_name_1, sheet_name_1, Hz_shaft[0])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_2 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum2.xlsx"   # y_1
sheet_name_2 = "Sensor 2"
frequency, amplitude = process_excel_data(file_name_2, sheet_name_2, Hz_shaft[0])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_3 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum3.xlsx"   # x_2
sheet_name_3 = "Sensor 3"
frequency, amplitude = process_excel_data(file_name_3, sheet_name_3, Hz_shaft[1])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")    # y_2
file_name_4 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum4.xlsx"
sheet_name_4 = "Sensor 4"
frequency, amplitude = process_excel_data(file_name_4, sheet_name_4, Hz_shaft[1])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")
print("---------------------------------------------")

a_x1 =data_amplitude[0]
a_y1 =data_amplitude[1]
a_x2 =data_amplitude[2]
a_y2 =data_amplitude[3]
print("--------------------0 th Measurements-----------------------")
print(f"{"Amplitude of x_1:"}",a_x1,f"\n{"Amplitude of y_1:"}",a_y1,f"\n{"Amplitude of x_2:"}",a_x2,f"\n{"Amplitude of y_2:"}",a_y2)
print("--------------------------------------------------------------------------------------------------------------")


print("---------------------------------------------")
file_name_11 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum11.xlsx" # x_1
sheet_name_11 = "Sensor 1"
frequency, amplitude = process_excel_data(file_name_11, sheet_name_11, Hz_shaft[2])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_21 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum21.xlsx"   # y_1
sheet_name_21 = "Sensor 2"
frequency, amplitude = process_excel_data(file_name_21, sheet_name_21, Hz_shaft[2])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_31 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum31.xlsx"   # x_2
sheet_name_31 = "Sensor 3"
frequency, amplitude = process_excel_data(file_name_31, sheet_name_31, Hz_shaft[3])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")    # y_2
file_name_41 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum41.xlsx"
sheet_name_41 = "Sensor 4"
frequency, amplitude = process_excel_data(file_name_41, sheet_name_41, Hz_shaft[3])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")
print("---------------------------------------------")

a_x11 =data_amplitude[4]
a_y11 =data_amplitude[5]
a_x12 =data_amplitude[6]
a_y12 =data_amplitude[7]
print("--------------------1 st Measurements-----------------------")
print(f"{"Amplitude of x_11:"}",a_x11,f"\n{"Amplitude of y_11:"}",a_y11,f"\n{"Amplitude of x_12:"}",a_x12,f"\n{"Amplitude of y_12:"}",a_y12)
print("--------------------------------------------------------------------------------------------------------------")



print("---------------------------------------------")
file_name_12 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum12.xlsx" # x_1
sheet_name_12 = "Sensor 1"
frequency, amplitude = process_excel_data(file_name_12, sheet_name_12, Hz_shaft[4])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_22 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum22.xlsx"   # y_1
sheet_name_22 = "Sensor 2"
frequency, amplitude = process_excel_data(file_name_22, sheet_name_22, Hz_shaft[4])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_32 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum32.xlsx"   # x_2
sheet_name_32 = "Sensor 3"
frequency, amplitude = process_excel_data(file_name_32, sheet_name_32, Hz_shaft[5])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")

print("---------------------------------------------")
file_name_42 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\Spectrum42.xlsx" # y_2
sheet_name_42 = "Sensor 4"
frequency, amplitude = process_excel_data(file_name_42, sheet_name_42, Hz_shaft[5])
data_amplitude.append(round(float(amplitude),8))
data_frequency.append(round(float(frequency),8))
if frequency is not None:
    print(f"Frequency: {frequency:.8f}, Amplitude: {amplitude:.8f}")
print("---------------------------------------------")

a_x21 =data_amplitude[8]
a_y21 =data_amplitude[9]
a_x22 =data_amplitude[10]
a_y22 =data_amplitude[11]
print("--------------------2 st Measurements-----------------------")
print(f"{"Amplitude of x_21:"}",a_x21,f"\n{"Amplitude of y_21:"}",a_y21,f"\n{"Amplitude of x_22:"}",a_x22,f"\n{"Amplitude of y_22:"}",a_y22)
print("--------------------------------------------------------------------------------------------------------------")

print("################################################################################################################")

#dt_phase_list.insert(3,25)  # çıkardığımız phase angle ı burada listeye ekliyor.
#period_list.insert(3,0.057)

# Use the trial weights as Wcal1 and Wcal2
Wcal1 = float(data.get("Trial Weight 1", 0)) # Default to 0 if not found
Wcal2 = float(data.get("Trial Weight 2", 0))  # Default to 0 if not found
print(f"Wcal1: {Wcal1}, Wcal2: {Wcal2}")

A1 = np.sqrt(a_x1**2+a_y1**2) # baslangıc_olcumu_duzlem1_amplitude
A2 = np.sqrt(a_x2**2+a_y2**2) # baslangıc_olcumu_duzlem2_amplitude
A11 = np.sqrt(a_x11**2+a_y11**2)#deneme_ağırlığı1_düzlem1_olcumu_amplitude
A12 = np.sqrt(a_x12**2+a_y12**2)#deneme_ağırlığı1_düzlem2_olcumu_amplitude
A21 = np.sqrt(a_x21**2+a_y21**2)#deneme_ağırlığı2_düzlem1_olcumu_amplitude
A22 = np.sqrt(a_x22**2+a_y22**2) #deneme_ağırlığı2_düzlem2_olcumu_amplitude
print("dt_phase_list:",dt_phase_list)
theta1=dt_phase_list[0]  # baslangıc_olcumu_duzlem1_derece
theta2=dt_phase_list[1]  # baslangıc_olcumu_duzlem2_derece
theta11=dt_phase_list[2] # deneme_ağırlığı1_düzlem1_olcumu_derece
theta12=dt_phase_list[3] # deneme_ağırlığı1_duzlem2_olcumu_derece
theta21=dt_phase_list[4] # deneme_ağırlığı2_duzlem1_olcumu_derece
theta22=dt_phase_list[5] # deneme_ağırlığı2_duzlem2_olcumu_derece

theta1_rad = cmath.pi * theta1 / 180
theta2_rad = cmath.pi * theta2 / 180
theta11_rad = cmath.pi * theta11 / 180
theta12_rad = cmath.pi * theta12 / 180
theta21_rad = cmath.pi * theta21 / 180
theta22_rad = cmath.pi * theta22 / 180

# Kompleks vektörler

baslangic_olcumu_duzlem1 = A1 * (cmath.cos(theta1_rad) + 1j * cmath.sin(theta1_rad))
baslangic_olcumu_duzlem2 = A2 * (cmath.cos(theta2_rad) + 1j * cmath.sin(theta2_rad))

deneme_agirligi1_duzlem1_olcumu = A11 * (cmath.cos(theta11_rad) + 1j * cmath.sin(theta11_rad))
deneme_agirligi1_duzlem2_olcumu = A12 * (cmath.cos(theta12_rad) + 1j * cmath.sin(theta12_rad))
deneme_agirligi2_duzlem1_olcumu = A21 * (cmath.cos(theta21_rad) + 1j * cmath.sin(theta21_rad))
deneme_agirligi2_duzlem2_olcumu = A22 * (cmath.cos(theta22_rad) + 1j * cmath.sin(theta22_rad))

degisim_vektoru11 = deneme_agirligi1_duzlem1_olcumu - baslangic_olcumu_duzlem1 #Vd11
degisim_vektoru12 = deneme_agirligi2_duzlem1_olcumu - baslangic_olcumu_duzlem1 #Vd12
degisim_vektoru21 = deneme_agirligi1_duzlem2_olcumu - baslangic_olcumu_duzlem2 #Vd21
degisim_vektoru22 = deneme_agirligi2_duzlem2_olcumu - baslangic_olcumu_duzlem2 #Vd22

etki_vektoru11 = (degisim_vektoru11) / (Wcal1)
etki_vektoru12 = (degisim_vektoru12) / (Wcal1)
etki_vektoru21 = (degisim_vektoru21) / (Wcal2)
etki_vektoru22 = (degisim_vektoru22) / (Wcal2)

# 2x2 kare matris
matrix_2x2 = np.array([[etki_vektoru11, etki_vektoru12],
                       [etki_vektoru21,  etki_vektoru22]])

# 2x1 matris (vektör)
matrix_2x1 = np.array([[baslangic_olcumu_duzlem1],
                       [baslangic_olcumu_duzlem2]])

try:
    inverse_matrix = np.linalg.inv(matrix_2x2)
    print("2x2 Matrisin Tersi:\n", inverse_matrix)
except np.linalg.LinAlgError:
    print("2x2 matris terslenemez (determinant sıfır olabilir).")
    inverse_matrix = None

# Eğer tersi varsa, 2x2 ters matris ile 2x1 matris çarpımı
if inverse_matrix is not None:
    result = np.dot(inverse_matrix, matrix_2x1)
    print("\nSonuç (2x2 Ters Matris * 2x1 Matris):\n", result)

    Wbal1, Wbal2 = result[0, 0], result[1, 0]

    magnitude_Wbal1 = np.sqrt(Wbal1.real**2 + Wbal1.imag**2)
    magnitude_Wbal2 = np.sqrt(Wbal2.real**2 + Wbal2.imag**2)
    print("magnitude_Wbal1",magnitude_Wbal1)
    print("magnitude_Wbal2", magnitude_Wbal2)
    ratio_Wbal1 = Wbal1.imag / Wbal1.real
    ratio_Wbal2 = Wbal2.imag / Wbal2.real

    angle_Wbal1_rad = np.arctan(ratio_Wbal1)  # Radyan
    angle_Wbal2_rad = np.arctan(ratio_Wbal2)  # Radyan

    angle_Wbal1_deg = np.degrees(angle_Wbal1_rad)  # Derece
    angle_Wbal2_deg = np.degrees(angle_Wbal2_rad)  # Derece

    print(f"Wbal1 Açısı (Radyan): {angle_Wbal1_rad}")
    print(f"Wbal1 Açısı (Derece): {angle_Wbal1_deg}")

    print(f"Wbal2 Açısı (Radyan): {angle_Wbal2_rad}")
    print(f"Wbal2 Açısı (Derece): {angle_Wbal2_deg}")

    # ✅ Move this inside the block
    if magnitude_Wbal1 and magnitude_Wbal2 and angle_Wbal1_deg and angle_Wbal2_deg and A1 and A11:
        result_data = {
            "magnitude_Wbal1": magnitude_Wbal1,
            "magnitude_Wbal2": magnitude_Wbal2,
            "angle_Wbal1_deg": angle_Wbal1_deg,
            "angle_Wbal2_deg": angle_Wbal2_deg,
            "A1": A1,
            "A11": A11,
            "a_x1": a_x1,
            "a_x2": a_x2,
            "a_y1": a_y1,
            "a_y2": a_y2,
            "a_x11": a_x11,
            "a_x12": a_x12,
            "a_y11": a_y11,
            "a_y12": a_y12,
            "a_x21": a_x21,
            "a_x22": a_x22,
            "a_y21": a_y21,
            "a_y22": a_y22

        }
        with open(DATA_FILE_2, "w") as f:
            json.dump(result_data, f, indent=4)
        print("\n✅ Data saved successfully to data2.json!")
    else:
        print("\n⚠ Data not saved due to calculation error.")