REM activate venv
..\api\onnx_env\Scripts\Activate.bat

REM build bundle
pyinstaller win10.directml.dir.spec

REM add additional files
xcopy \gfpgan \dist\server\gfpgan /t /e
xcopy ..\api\gui \dist\client /t /e
xcopy ..\api\schemas \dist\schemas /t /e
xcopy ..\api\logging.yaml \dist\logging.yaml
xcopy ..\api\params.json \dist\params.json
xcopy ..\docs \dist\docs /t /e
xcopy ..\models \dist\models /t /e
xcopy ..\outputs \dist\outputs /t /e

REM get commit info
git rev-parse HEAD > commit.txt
set /p GIT_SHA=<commit.txt

REM create archive
7za a ..\dist\onnx-web-v0.10.0-rc-%GIT_SHA%.zip "\dist\*"
