apt-get update
apt-get install zip
unzip apex-master.zip
cd apex-master || exit
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
