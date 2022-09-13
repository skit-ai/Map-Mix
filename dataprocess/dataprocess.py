import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm 

BASE_PATH = "/root/LRE2017Dataset/LRE2017_16k/"
TARGET_SR = 16000

# TEST SET
df = pd.read_csv("/root/LRE2017Dataset/LRE2017/lre17-test-set-less-than-60.csv")
save_path = "/root/LRE2017Dataset/LRE2017_16k/eval/"
csv_name = "testset.csv"

# df = pd.read_csv("/root/LRE2017Dataset/LRE2017/lre17-val-set-less-than-60.csv")
# save_path = "/root/LRE2017Dataset/LRE2017_16k/dev/"
# csv_name = "valset.csv"

# df = pd.read_csv("/root/LRE2017Dataset/LRE2017/lre17-segmented-train-set-5-hour-each-set-3.csv")
# save_path = "/root/LRE2017Dataset/LRE2017_16k/train/set3/"
# csv_name = "trainset3.csv"

for i in tqdm(range(len(df))):
    file_name = df['audiopath'][i]
    wav, sr = librosa.load(file_name)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    out_file = save_path + file_name.split("/")[-1].split(".")[0]+".wav"
    sf.write(out_file, wav, TARGET_SR)


df["audiopath"] = save_path + df["audiopath"].str.split("/").str[-1]
df.to_csv(BASE_PATH+csv_name)
print(df["audiopath"].head()) 



# df = "/root/LRE2017Dataset/LRE2017/lre17-val-set-less-than-60.csv"
# save_path = "/root/LRE2017Dataset/LRE2017_16k/dev/"

# df = pd.read_csv("/root/LRE2017Dataset/LRE2017/lre17-segmented-train-set-5-hour-each-set-1.csv")
# save_path = "/root/LRE2017Dataset/LRE2017_16k/train/set1/"