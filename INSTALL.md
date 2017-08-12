### summary
- 소스와 데이터 repository 는 각각 분리되어 있습니다.
- 데이터 repository는 git lfs를 사용하여, 설치가 필요합니다. 
    - https://github.com/git-lfs/git-lfs/wiki/Installation
    - SourceTree 등의 tool을 이용하시면, git lfs 파일의 clone이 쉽습니다.
```shell
# OSX 기준 (최초 설치)
brew install git git-lfs
git lfs install
```
- 개발 환경
    - 서버: Ubuntu 16.04 + Anaconda3 4.3 + Python 3.5 + Tensorflow-GPU 1.2
    - 로컬: OSX 10.12 + Anaconda3 4.3 + Python 3.5 + Tensorflow-CPU 1.2

### install source codes
- anaconda3 설치 https://www.continuum.io/downloads
- `pip install tensorflow-gpu`
    - 또는 `pip install tensorflow-cpu`
- git clone 실행하는 디렉토리는 `~/workspace` 입니다.
- 원할한 소스 실행을 위해서는 환경변수 PYTHONPATH 에 `~/workspace/nlp4kor` 를 추가하셔야 합니다.
    - `export PYTHONPATH=~/workspace/nlp4kor:$PYTHONPATH`
    - ~/.bash_profile 또는 ~/.profile 에 한 줄을 추가해 주세요.
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
git clone https://gitlab.com/bage79/nlp4kor-mnist.git
git clone https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org.git
```