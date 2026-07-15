import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Library audio processing
import librosa
import librosa.display

class AudioFeatureExtractor:
    """Ekstraktor fitur audio untuk dataset tanpa label"""

    def __init__(self, sr=22050):
        self.sr = sr
        self.features_list = []

    def discover_audio_files(self, folder_path):
        """Menemukan semua file audio dalam folder dengan penanganan error"""
        audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')
        audio_files = []

        print(f"🔍 Mencari file audio di: {folder_path}")

        # Validasi apakah folder exists
        if not os.path.exists(folder_path):
            print(f"❌ ERROR: Folder '{folder_path}' tidak ditemukan!")
            return []
        
        if not os.path.isdir(folder_path):
            print(f"❌ ERROR: '{folder_path}' bukan folder!")
            return []

        try:
            # Dapatkan semua file di folder
            all_files = os.listdir(folder_path)
            
            if not all_files:
                print("❌ Folder kosong!")
                return []

            for file in all_files:
                file_lower = file.lower()
                if file_lower.endswith(audio_extensions):
                    full_path = os.path.join(folder_path, file)
                    
                    # Validasi file exists dan bisa diakses
                    if not os.path.isfile(full_path):
                        continue
                    
                    try:
                        file_size = os.path.getsize(full_path)
                        # Skip file yang terlalu kecil (kurang dari 10KB)
                        if file_size < 10240:  # 10KB
                            print(f"⚠️  File {file} terlalu kecil, dilewati")
                            continue
                            
                        audio_files.append({
                            'file_path': full_path,
                            'filename': file,
                            'file_size': file_size
                        })
                    except (OSError, PermissionError) as e:
                        print(f"⚠️  Tidak bisa mengakses {file}: {str(e)}")
                        continue

            print(f"🎵 Ditemukan {len(audio_files)} file audio yang valid")
            
            # Tampilkan detail file yang ditemukan
            if audio_files:
                total_size = sum(f['file_size'] for f in audio_files) / (1024*1024)  # MB
                print(f"📦 Total ukuran: {total_size:.2f} MB")
                
                # Tampilkan 5 file pertama
                print("\n📋 Sample file yang ditemukan:")
                for i, audio_file in enumerate(audio_files[:5]):
                    size_mb = audio_file['file_size'] / (1024*1024)
                    print(f"   {i+1}. {audio_file['filename']} ({size_mb:.2f} MB)")
                
                if len(audio_files) > 5:
                    print(f"   ... dan {len(audio_files) - 5} file lainnya")
            
            return audio_files

        except PermissionError:
            print(f"❌ ERROR: Tidak ada izin untuk mengakses folder '{folder_path}'")
            return []
        except Exception as e:
            print(f"❌ ERROR: Terjadi kesalahan saat membaca folder: {str(e)}")
            return []

    def extract_basic_features(self, file_path):
        """Ekstrak fitur dasar dengan error handling yang lebih baik"""
        try:
            # Load audio file dengan timeout implicit melalui duration
            y, sr = librosa.load(file_path, sr=self.sr, duration=300, mono=True)
            
            features = {}
            
            # 1. BASIC TEMPORAL FEATURES
            features['duration'] = len(y) / sr
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 2. RHYTHM FEATURES
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['n_beats'] = len(beats) if beats is not None else 0
            except:
                features['tempo'] = 0
                features['n_beats'] = 0
            
            # 3. SPECTRAL FEATURES
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # 4. MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
            # 5. CHROMA FEATURES
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_stft_mean'] = np.mean(chroma_stft)
            features['chroma_stft_std'] = np.std(chroma_stft)
            
            return features
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
            return None

    def process_folder(self, folder_path, output_file='audio_features.csv'):
        """Process seluruh folder dan simpan ke CSV"""
        print("🚀 MEMULAI EKSTRAKSI FITUR AUDIO")
        print("=" * 50)
        
        # Temukan file audio
        audio_files = self.discover_audio_files(folder_path)
        
        if not audio_files:
            print("❌ Tidak ada file audio yang dapat diproses!")
            return None
        
        all_features = []
        success_count = 0
        fail_count = 0
        
        print(f"\n📊 Memproses {len(audio_files)} file...")
        
        for audio_info in tqdm(audio_files, desc="Ekstraksi Fitur"):
            features = self.extract_basic_features(audio_info['file_path'])
            
            if features is not None:
                # Tambahkan metadata
                features['filename'] = audio_info['filename']
                features['file_path'] = audio_info['file_path']
                features['file_size_mb'] = audio_info['file_size'] / (1024*1024)
                
                all_features.append(features)
                success_count += 1
            else:
                fail_count += 1
        
        # Buat DataFrame dan simpan
        if all_features:
            df = pd.DataFrame(all_features)
            
            # Reorder columns untuk membuat filename pertama
            cols = ['filename', 'file_path', 'file_size_mb'] + [col for col in df.columns if col not in ['filename', 'file_path', 'file_size_mb']]
            df = df[cols]
            
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ EKSTRAKSI SELESAI!")
            print(f"📊 File berhasil: {success_count}")
            print(f"❌ File gagal: {fail_count}")
            print(f"💾 Dataset disimpan: {output_file}")
            print(f"🎯 Total fitur: {len(df.columns)}")
            
            # Tampilkan preview
            print(f"\n📋 PREVIEW DATA:")
            print(df.head(3).to_string(max_cols=8, index=False))
            
            return df
        else:
            print("❌ Tidak ada fitur yang berhasil diekstraksi!")
            return None

# FUNGSI UTAMA YANG MUDAH DIGUNAKAN
def main():
    """Program utama yang user-friendly"""
    print("🎵 EKSTRAKSI FITUR AUDIO OTOMATIS")
    print("=" * 40)
    
    # Dapatkan path folder secara interaktif
    while True:
        folder_path = input("\n📁 Masukkan path folder berisi file audio: ").strip()
        
        # Hilangkan quotes jika ada
        folder_path = folder_path.strip('"\'')
        
        if not folder_path:
            print("❌ Path tidak boleh kosong!")
            continue
            
        if folder_path.lower() == 'current':
            folder_path = '.'
        
        # Validasi folder
        if not os.path.exists(folder_path):
            print("❌ Folder tidak ditemukan!")
            print("💡 Tips: Gunakan path lengkap atau 'current' untuk folder saat ini")
            continue
            
        if not os.path.isdir(folder_path):
            print("❌ Path yang diberikan bukan folder!")
            continue
            
        break
    
    # Konfirmasi dengan user
    file_count = len([f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac'))])
    
    if file_count == 0:
        print("❌ Tidak ditemukan file audio di folder tersebut!")
        print("💡 Format yang didukung: MP3, WAV, FLAC, M4A, AAC")
        return
    
    print(f"🎵 Akan memproses {file_count} file audio")
    
    confirm = input("\n🚀 Lanjutkan ekstraksi? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Dibatalkan oleh user")
        return
    
    # Jalankan ekstraksi
    extractor = AudioFeatureExtractor()
    df = extractor.process_folder(folder_path)
    
    if df is not None:
        print(f"\n🎉 SUKSES! Dataset siap untuk preprocessing ML.")
        print("💡 Selanjutnya: Gunakan file CSV untuk analisis dan modeling")

# FUNGSI UNTUK TEST CEPAT
def quick_test():
    """Test cepat dengan folder saat ini"""
    print("🧪 QUICK TEST - Folder Saat Ini")
    
    extractor = AudioFeatureExtractor()
    
    # Cek folder saat ini
    current_dir = '.'
    audio_files = extractor.discover_audio_files(current_dir)
    
    if not audio_files:
        print("\n❌ Tidak ada file audio yang ditemukan!")
        print("\n💡 CARA MENAMBAH FILE AUDIO:")
        print("1. Copy file MP3/WAV ke folder ini:")
        print(f"   {os.path.abspath('.')}")
        print("2. Atau jalankan program dengan path folder lain")
        return
    
    print(f"\n🎵 Ditemukan {len(audio_files)} file audio")
    
    # Tanya user
    confirm = input("🚀 Jalankan ekstraksi? (y/n): ").lower().strip()
    if confirm == 'y':
        df = extractor.process_folder(current_dir, 'quick_test_features.csv')
    else:
        print("❌ Dibatalkan")

if __name__ == "__main__":
    print("Pilih mode:")
    print("1. Ekstraksi dari folder tertentu")
    print("2. Quick test (folder saat ini)")
    
    choice = input("Masukkan pilihan (1/2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    else:
        print("❌ Pilihan tidak valid! Menjalankan quick test...")
        quick_test()