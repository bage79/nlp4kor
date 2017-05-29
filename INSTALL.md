### summary
- git 디렉토리는 `~/workspace` 입니다.
- 소스와 데이터 repository 는 각각 분리되어 있습니다.
- 데이터 repository는 git lfs를 사용하여, 설치가 필요합니다. https://github.com/git-lfs/git-lfs/wiki/Installation
- 원할한 소스 실행을 위해서는 환경변수 PYTHONPATH 에 `~/workspace/nlp4kor` 를 추가하셔야 합니다. 
- 모든 명령은 Ubuntu를 기준으로 합니다. (대부분 OSX 호환)

### install source codes
```shell
mkdir ~/workspae
cd ~/workspace
git clone https://github.com/bage79/nlp4kor.git
export PYTHONPATH=~/workspace/nlp4kor:$PYTHONPATH # ~/.bash_profile 또는 ~/.bashrc에 입력시 생략
```

### download datasets
```shell
cd ~/workspace
git clone https://bitbucket.org/bage79/nlp4kor-mnist
git clone https://bitbucket.org/bage79/nlp4kor-ko.wikipedia.org
```