### summary
- 소스와 데이터 repository 는 각각 분리되어 있습니다.
- 데이터 repository는 git lfs를 사용하여, 설치가 필요합니다. 
    - https://github.com/git-lfs/git-lfs/wiki/Installation
```shell
# OSX 기준 (최초 설치)
brew install git git-lfs
git lfs install
```
- 개발 환경
    - 서버: Ubuntu 16.04 + Anaconda3 4.3 + Python 3.5 + Tensorflow-GPU 1.1
    - 로컬: OSX 10.12 + Anaconda3 4.3 + Python 3.5 + Tensorflow-CPU 1.1

### install source codes
- anaconda3 설치 https://www.continuum.io/downloads
- `pip install tensorflow-gpu`
    - 또는 `pip install tensorflow-cpu`
- git clone 되는 상위 디렉토리는 `~/workspace` 입니다.
- 원할한 소스 실행을 위해서는 환경변수 PYTHONPATH 에 `~/workspace/nlp4kor` 를 추가하셔야 합니다.
    - `export PYTHONPATH=~/workspace/nlp4kor:$PYTHONPATH`
    - ~/.bash_profile 또는 ~/.bashrc에 한 줄을 추가해 주세요.
```shell
mkdir ~/workspae
cd ~/workspace
git clone https://github.com/bage79/nlp4kor.git
cd ~/workspace/nlp4kor
pip install -r requirements.txt
```

### download datasets
```shell
cd ~/workspace
git clone https://bitbucket.org/bage79/nlp4kor-mnist
git clone https://bitbucket.org/bage79/nlp4kor-ko.wikipedia.org
```