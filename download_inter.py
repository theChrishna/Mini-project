import os
import urllib.request
import zipfile

os.makedirs('fonts', exist_ok=True)
url = 'https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip'
zip_path = 'fonts/Inter.zip'

print("Downloading Inter font...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
    out_file.write(response.read())

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('fonts/')

print("Done. Fonts available in fonts/ folder.")
