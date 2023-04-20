import subprocess
import os
import requests
import zipfile
import shutil

# install myo sdk
print("Installing MYO SDK...")
os.mkdir("temp")

url = 'https://github.com/NiklasRosenstein/myo-python/releases/download/v1.0.4/myo-sdk-win-0.9.0.zip'
r = requests.get(url, allow_redirects=True)
open('temp/myo-sdk-win-0.9.0.zip', 'wb').write(r.content)

with zipfile.ZipFile('temp/myo-sdk-win-0.9.0.zip', 'r') as zip_ref:
    zip_ref.extractall("C:/")

# download myo installer
print("Downloading MYO Installer...")
url = 'https://github.com/NiklasRosenstein/myo-python/releases/download/v1.0.4/Myo+Connect+Installer.exe'
r = requests.get(url, allow_redirects=True)
open('temp/Myo+Connect+Installer.exe', 'wb').write(r.content)

path = os.path.abspath("temp/Myo+Connect+Installer.exe")
print(path)
subprocess.run(r"{path}".format(path=path), shell=True)

shutil.rmtree("temp", ignore_errors=True)

subprocess.run("conda create --name GR_env --file requirements.txt")

print("Done!")